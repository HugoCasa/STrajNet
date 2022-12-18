import tensorflow as tf
import numpy as np
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils.occupancy_flow_renderer import _SampledPoints

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2
_ObjectType = scenario_pb2.Track.ObjectType

import tensorflow_addons as tfa

from typing import List, Mapping, Sequence

from trajLoss import min_ade, torch_gather

import math


ogm_config = []


class ConsistencyLoss(tf.keras.losses.Loss):
    def __init__(self, use_focal_loss = False, replica = 1):

        self.use_focal_loss = use_focal_loss
        self.focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        self.replica = replica

        a = tf.concat([tf.random.uniform([4, 48, 6, 8, 3]), tf.zeros([4,16,6,8,3])], axis=1)

        probs = tf.concat([tf.ones([4, 64, 1]) * -0.02, tf.ones([4,64,5]) * -2], axis=2)

        dummy_preds = a, probs

        dummy_gt = a[:,:,0,:,:]
        gt_masks = tf.concat([tf.ones([4, 16, 8]), tf.zeros([4,48,8])], axis=1)

        dummy_ogm = [tf.zeros([4,256,256]) for _ in range(8)]
        dummy_flow = [tf.zeros([4,256,256,2]) for _ in range(8)]
        dummy_ogm_flow = occupancy_flow_grids.WaypointGrids()
        dummy_ogm_flow.vehicles.observed_occupancy = dummy_ogm
        dummy_ogm_flow.vehicles.occluded_occupancy = dummy_ogm
        dummy_ogm_flow.vehicles.flow = dummy_flow

        gt_refs = tf.ones([4, 64, 4])
        gt_infos = tf.ones([4, 64, 3])
        loss_dict = self(dummy_preds, dummy_ogm_flow, dummy_gt, gt_masks, gt_refs, gt_infos)
        print(loss_dict)


    def __call__(self, traj_preds_and_probs, ogm_flow: occupancy_flow_grids.WaypointGrids, gt_trajs, gt_masks, gt_refs, gt_infos):

        # traj_preds_and_probs: (preds, probs)
        # preds (B, 64, 6, 8, 3)
        # probs (B, 64, 6)

        # gt_trajs (B, 64, 8, 3)
        # gt_masks (B, 64, 8)
        # gt_refs (B, 64, 4) (x,y,yaw,valid) in global
        # gt_infos (B, 64, 3) (type, length, width)
        gt_refs_valid = gt_refs[:, :, 3] # (B, 64)
        gt_refs = gt_refs[:, :, :3] # (B, 64, 3)


        # Unpack arguments
        trajs, log_probs = traj_preds_and_probs # (B, 64, 6, 8, 3), (B, 64, 6)

        # take closest mode to ground truth
        errs, inds = min_ade(trajs, gt_trajs, gt_masks)
        inds = tf.tile(inds[:, :, tf.newaxis, tf.newaxis, tf.newaxis],[1,1,1,8,3]) # (B, 64, 1, 8, 3)
        trajs = tf.squeeze(torch_gather(trajs, inds, 2), axis=2) # (B, 64, 8, 3)

        # shift predicted trajs into global ref (SDC)

        gt_refs_expanded = tf.tile(gt_refs[:, :, tf.newaxis, :], [1,1,8,1]) # (B, 64, 8, 3)

        ref_x, ref_y, ref_yaw = tf.split(gt_refs_expanded, 3, axis=3)
        x, y, yaw = tf.split(trajs, 3, axis=3)

        angle = ref_yaw - math.pi / 2

        tx = tf.cos(angle) * x - tf.sin(angle) * y
        ty = tf.sin(angle) * x + tf.cos(angle) * y
        x, y = tx, ty

        yaw = yaw + angle

        x = x + ref_x
        y = y + ref_y

        trajs = tf.squeeze(tf.stack([x,y,yaw], axis=3), axis=4) # (B, 64, 8, 3)


        # construct occupancy grid from trajectories
        config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        config_text = """
        grid_height_cells: 256
        grid_width_cells: 256
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(config_text, config)
        occupancy_inputs = {
            'trajs': trajs,
            'gt_masks': gt_masks[:, :, :, tf.newaxis], # (B, 64, 8, 1)
            'gt_infos': tf.tile(gt_infos[:, :, tf.newaxis, :], [1,1,8,1]), # (B, 64, 8, 1)
        }
        trajs_occupancy = render_occupancy_from_inputs(occupancy_inputs, config) # (B, 256, 256, 8)

        flow_inputs = {
            'trajs': tf.concat([gt_refs[:, :, tf.newaxis, :], trajs], axis=2), # (B, 64, 9, 3)
            'gt_masks': tf.concat([gt_refs_valid[:, :, tf.newaxis], gt_masks], axis=2)[:, :, :, tf.newaxis], # (B, 64, 9, 1)
            'gt_infos': tf.tile(gt_infos[:, :, tf.newaxis, :], [1,1,9,1]), # (B, 64, 9, 1)
        }
        trajs_flow = render_flow_from_inputs(flow_inputs, config) # (B, 256, 256, 8, 2)

        # compare to predicted

        loss_dict = {
            'occupancy': [],
            'flow': []
        }

        for k in range(8):
            ogm = ogm_flow.vehicles.observed_occupancy[k] + ogm_flow.vehicles.occluded_occupancy[k]
            ogm = tf.clip_by_value(ogm, 0, 1)

            flow = ogm_flow.vehicles.flow[k]

  
            # Accumulate over waypoints.
            loss_dict['occupancy'].append(
                self._sigmoid_loss(
                    true_occupancy=trajs_occupancy[:, :, :, k],
                    pred_occupancy=ogm)
            ) 

            loss_dict['flow'].append(self._flow_loss(trajs_flow[:, :, :, k], flow))

        n_dict = {}
        n_dict['occupancy'] = tf.math.add_n(loss_dict['occupancy']) / 8
        n_dict['flow'] = tf.math.add_n(loss_dict['flow']) / 8

        return n_dict

    

    def _sigmoid_loss(
        self,
        true_occupancy: tf.Tensor,
        pred_occupancy: tf.Tensor,
        loss_weight: float = 1000,
    ) -> tf.Tensor:
        """Computes sigmoid cross-entropy loss over all grid cells."""
        # Since the mean over per-pixel cross-entropy values can get very small,
        # we compute the sum and multiply it by the loss weight before computing
        # the mean.
        if self.use_focal_loss:
            xe_sum = tf.reduce_sum(
                self.focal_loss(
                    y_true=self._batch_flatten(true_occupancy),
                    y_pred=self._batch_flatten(pred_occupancy)
                )) + tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        else:
            xe_sum = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        # Return mean.
        return loss_weight * xe_sum / (tf.size(pred_occupancy, out_type=tf.float32)*self.replica)

    def _flow_loss(
        self,
        true_flow: tf.Tensor,
        pred_flow: tf.Tensor,
        loss_weight: float = 1,
    ) -> tf.Tensor:
        """Computes L1 flow loss."""
        diff = true_flow - pred_flow
        # Ignore predictions in areas where ground-truth flow is zero.
        # [batch_size, height, width, 1], [batch_size, height, width, 1]
        true_flow_dx, true_flow_dy = tf.split(true_flow, 2, axis=-1)
        # [batch_size, height, width, 1]
        flow_exists = tf.logical_or(
            tf.not_equal(true_flow_dx, 0.0),
            tf.not_equal(true_flow_dy, 0.0),
        )
        flow_exists = tf.cast(flow_exists, tf.float32)
        diff = diff * flow_exists
        diff_norm = tf.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
        mean_diff = tf.math.divide_no_nan(
            tf.reduce_sum(diff_norm),
            (tf.reduce_sum(flow_exists)*self.replica / 2))  # / 2 since (dx, dy) is counted twice.
        return loss_weight * mean_diff

    def _batch_flatten(self,input_tensor: tf.Tensor) -> tf.Tensor:
        """Flatten tensor to a shape [batch_size, -1]."""
        image_shape = tf.shape(input_tensor)
        return tf.reshape(input_tensor, tf.concat([image_shape[0:1], [-1]], 0))

def render_occupancy_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> occupancy_flow_data.AgentGrids:
    """Creates topdown renders of agents grouped by agent class.

    Renders agent boxes by densely sampling points from their boxes.
    """
    sampled_points = _sample_and_filter_agent_points(
        inputs=inputs,
        config=config,
    )

    agent_x = sampled_points.x
    agent_y = sampled_points.y
    agent_type = sampled_points.agent_type
    agent_valid = sampled_points.valid

    # Set up assert_shapes.
    assert_shapes = tf.debugging.assert_shapes
    batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
    topdown_shape = [
        batch_size, config.grid_height_cells, config.grid_width_cells, num_steps
    ]

    # Transform from world coordinates to topdown image coordinates.
    # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
    agent_x, agent_y, point_is_in_fov = occupancy_flow_renderer._transform_to_image_coordinates(
        points_x=agent_x,
        points_y=agent_y,
        config=config,
    )
    assert_shapes([(point_is_in_fov,
                    [batch_size, num_agents, num_steps, points_per_agent])])

    # Filter out points from invalid objects.
    agent_valid = tf.cast(agent_valid, tf.bool)
    point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

    # Only keep vehicles
    object_type = _ObjectType.TYPE_VEHICLE

    # Collect points for agent type (vehicles)
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                        agent_type_matches)

    assert_shapes([
        (should_render_point,
        [batch_size, num_agents, num_steps, points_per_agent]),
    ])

    # Scatter points across topdown maps for each timestep.  The tensor
    # `point_indices` holds the indices where `should_render_point` is True.
    # It is a 2-D tensor with shape [n, 4], where n is the number of valid
    # agent points inside FOV.  Each row in this tensor contains indices over
    # the following 4 dimensions: (batch, agent, timestep, point).

    # [num_points_to_render, 4]
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)
    # [num_points_to_render, 1]
    x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]
    assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                (y_img_coord, [num_points_to_render, 1])])

    # [num_points_to_render, 4]
    xy_img_coord = tf.concat(
        [
            point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
            point_indices[:, 2:3],
        ],
        axis=1,
    )
    # [num_points_to_render]
    gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

    # [batch_size, grid_height_cells, grid_width_cells, num_steps]
    topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
    assert_shapes([(topdown, topdown_shape)])

    # scatter_nd() accumulates values if there are repeated indices.  Since
    # we sample densely, this happens all the time.  Clip the final values.
    topdown = tf.clip_by_value(topdown, 0.0, 1.0)

    return topdown


def _sample_and_filter_agent_points(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> _SampledPoints:
    """Samples points and filters them according to current visibility of agents.

    Args:
        inputs: Dict of input tensors from the motion dataset.
        config: OccupancyFlowTaskConfig proto message.

    Returns:
        _SampledPoints: containing x, y, z coordinates, type, and valid bits.
    """
    # Set up assert_shapes.
    assert_shapes = tf.debugging.assert_shapes
    batch_size, num_agents, num_steps, _ = (inputs['trajs'].shape.as_list())
    points_per_agent = (
        config.agent_points_per_side_length * config.agent_points_per_side_width)

    # Sample points from agent boxes over specified time frames.
    # All fields have shape [batch_size, num_agents, num_steps, points_per_agent].
    sampled_points = _sample_agent_points(
        inputs,
        points_per_side_length=config.agent_points_per_side_length,
        points_per_side_width=config.agent_points_per_side_width,
    )

    field_shape = [batch_size, num_agents, num_steps, points_per_agent]
    assert_shapes([
        (sampled_points.x, field_shape),
        (sampled_points.y, field_shape),
        (sampled_points.valid, field_shape),
        (sampled_points.agent_type, field_shape),
    ])

    agent_valid = tf.cast(sampled_points.valid, tf.bool)

    return _SampledPoints(
        x=sampled_points.x,
        y=sampled_points.y,
        z=None,
        agent_type=sampled_points.agent_type,
        valid=agent_valid,
    )


def _sample_agent_points(
    inputs: Mapping[str, tf.Tensor],
    points_per_side_length: int,
    points_per_side_width: int,
) -> _SampledPoints:
    """Creates a set of points to represent agents in the scene.

    For each timestep in `times`, samples the interior of each agent bounding box
    on a uniform grid to create a set of points representing the agent.

    Args:
        inputs: Dict of input tensors from the motion dataset.
        points_per_side_length: The number of points along the length of the agent.
        points_per_side_width: The number of points along the width of the agent.

    Returns:
        _SampledPoints object.
    """

    # All fields: [batch_size, num_agents, num_steps, 1].
    x, y, bbox_yaw =  tf.split(inputs['trajs'], 3, axis=3)
    agent_type, length, width = tf.split(inputs['gt_infos'], 3, axis=3)
    valid = inputs['gt_masks']
    shape = ['batch_size', 'num_agents', 'num_steps', 1]
    tf.debugging.assert_shapes([
        (x, shape),
        (y, shape),
        (bbox_yaw, shape),
        (length, shape),
        (width, shape),
        (valid, shape),
    ])


    return _sample_points_from_agent_boxes(
        x=x,
        y=y,
        bbox_yaw=bbox_yaw,
        width=width,
        length=length,
        agent_type=agent_type,
        valid=valid,
        points_per_side_length=points_per_side_length,
        points_per_side_width=points_per_side_width,
    )

def _sample_points_from_agent_boxes(
    x: tf.Tensor,
    y: tf.Tensor,
    bbox_yaw: tf.Tensor,
    width: tf.Tensor,
    length: tf.Tensor,
    agent_type: tf.Tensor,
    valid: tf.Tensor,
    points_per_side_length: int,
    points_per_side_width: int,
) -> _SampledPoints:
    """Create a set of 3D points by sampling the interior of agent boxes.

    For each state in the inputs, a set of points_per_side_length *
    points_per_side_width points are generated by sampling within each box.

    Args:
        x: Centers of agent boxes X [..., 1] (any shape with last dim 1).
        y: Centers of agent boxes Y [..., 1] (same shape as x).
        bbox_yaw: Agent box orientations [..., 1] (same shape as x).
        width : Widths of agent boxes [..., 1] (same shape as x).
        length: Lengths of agent boxes [..., 1] (same shape as x).
        agent_type: Types of agents [..., 1] (same shape as x).
        valid: Agent state valid flag [..., 1] (same shape as x).
        points_per_side_length: The number of points along the length of the agent.
        points_per_side_width: The number of points along the width of the agent.

    Returns:
        _SampledPoints object.
    """
    assert_shapes = tf.debugging.assert_shapes
    assert_shapes([(x, [..., 1])])
    x_shape = x.get_shape().as_list()
    assert_shapes([(y, x_shape), (bbox_yaw, x_shape),
                    (width, x_shape), (length, x_shape), (valid, x_shape)])
    if points_per_side_length < 1:
        raise ValueError('points_per_side_length must be >= 1')
    if points_per_side_width < 1:
        raise ValueError('points_per_side_width must be >= 1')

    # Create sample points on a unit square or boundary depending on flag.
    if points_per_side_length == 1:
        step_x = 0.0
    else:
        step_x = 1.0 / (points_per_side_length - 1)
    if points_per_side_width == 1:
        step_y = 0.0
    else:
        step_y = 1.0 / (points_per_side_width - 1)
    unit_x = []
    unit_y = []
    for xi in range(points_per_side_length):
        for yi in range(points_per_side_width):
            unit_x.append(xi * step_x - 0.5)
            unit_y.append(yi * step_y - 0.5)

    # Center unit_x and unit_y if there was only 1 point on those dimensions.
    if points_per_side_length == 1:
        unit_x = np.array(unit_x) + 0.5
    if points_per_side_width == 1:
        unit_y = np.array(unit_y) + 0.5

    unit_x = tf.convert_to_tensor(unit_x, tf.float32)
    unit_y = tf.convert_to_tensor(unit_y, tf.float32)

    num_points = points_per_side_length * points_per_side_width
    assert_shapes([(unit_x, [num_points]), (unit_y, [num_points])])

    # Transform the unit square points to agent dimensions and coordinate frames.
    sin_yaw = tf.sin(bbox_yaw)
    cos_yaw = tf.cos(bbox_yaw)

    # [..., num_points]
    tx = cos_yaw * length * unit_x - sin_yaw * width * unit_y + x
    ty = sin_yaw * length * unit_x + cos_yaw * width * unit_y + y

    points_shape = x_shape[:-1] + [num_points]
    assert_shapes([(tx, points_shape), (ty, points_shape)])
    agent_type = tf.broadcast_to(agent_type, tx.shape)
    valid = tf.broadcast_to(valid, tx.shape)

    return _SampledPoints(x=tx, y=ty, z=None, agent_type=agent_type, valid=valid)

def render_flow_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> occupancy_flow_data.AgentGrids:
    """Compute top-down flow between timesteps `waypoint_size` apart.

    Returns (dx, dy) for each timestep.

    Args:
        inputs: Dict of input tensors from the motion dataset.
        config: OccupancyFlowTaskConfig proto message.

    Returns:
        vehicles: [batch_size, height, width, num_flow_steps, 2] float32
    """
    sampled_points = _sample_and_filter_agent_points(
        inputs=inputs,
        config=config,
    )

    agent_x = sampled_points.x
    agent_y = sampled_points.y
    agent_type = sampled_points.agent_type
    agent_valid = sampled_points.valid

    # Set up assert_shapes.
    assert_shapes = tf.debugging.assert_shapes
    batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
    # The timestep distance between flow steps.
    waypoint_size = 1
    num_flow_steps = num_steps - waypoint_size
    topdown_shape = [
        batch_size, config.grid_height_cells, config.grid_width_cells,
        num_flow_steps
    ]

    # Transform from world coordinates to topdown image coordinates.
    # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
    agent_x, agent_y, point_is_in_fov = occupancy_flow_renderer._transform_to_image_coordinates(
        points_x=agent_x,
        points_y=agent_y,
        config=config,
    )
    assert_shapes([(point_is_in_fov,
                    [batch_size, num_agents, num_steps, points_per_agent])])

    # Filter out points from invalid objects.
    agent_valid = tf.cast(agent_valid, tf.bool)

    # Backward Flow.
    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    dx = agent_x[:, :, :-waypoint_size, :] - agent_x[:, :, waypoint_size:, :]
    dy = agent_y[:, :, :-waypoint_size, :] - agent_y[:, :, waypoint_size:, :]
    assert_shapes([
        (dx, [batch_size, num_agents, num_flow_steps, points_per_agent]),
        (dy, [batch_size, num_agents, num_flow_steps, points_per_agent]),
    ])

    # Adjust other fields as well to reduce from num_steps to num_flow_steps.
    # agent_x, agent_y: Use later timesteps since flow vectors go back in time.
    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    agent_x = agent_x[:, :, waypoint_size:, :]
    agent_y = agent_y[:, :, waypoint_size:, :]
    # agent_type: Use later timesteps since flow vectors go back in time.
    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    agent_type = agent_type[:, :, waypoint_size:, :]
    # point_is_in_fov: Use later timesteps since flow vectors go back in time.
    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    point_is_in_fov = point_is_in_fov[:, :, waypoint_size:, :]
    # agent_valid: And the two timesteps.  They both need to be valid.
    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    agent_valid = tf.logical_and(agent_valid[:, :, waypoint_size:, :],
                                agent_valid[:, :, :-waypoint_size, :])

    # [batch_size, num_agents, num_flow_steps, points_per_agent]
    point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

    # only render vehicles
    object_type = _ObjectType.TYPE_VEHICLE
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                        agent_type_matches)
    assert_shapes([
        (should_render_point,
        [batch_size, num_agents, num_flow_steps, points_per_agent]),
    ])

    # [batch_size, height, width, num_flow_steps, 2]
    flow = _render_flow_points_for_one_agent_type(
        agent_x=agent_x,
        agent_y=agent_y,
        dx=dx,
        dy=dy,
        should_render_point=should_render_point,
        topdown_shape=topdown_shape,
    )

    return flow


def _render_flow_points_for_one_agent_type(
    agent_x: tf.Tensor,
    agent_y: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    should_render_point: tf.Tensor,
    topdown_shape: List[int],
) -> tf.Tensor:
    """Renders topdown (dx, dy) flow for given agent points.

    Args:
        agent_x: [batch_size, num_agents, num_steps, points_per_agent].
        agent_y: [batch_size, num_agents, num_steps, points_per_agent].
        dx: [batch_size, num_agents, num_steps, points_per_agent].
        dy: [batch_size, num_agents, num_steps, points_per_agent].
        should_render_point: [batch_size, num_agents, num_steps, points_per_agent].
        topdown_shape: Shape of the output flow field.

    Returns:
        Rendered flow as [batch_size, height, width, num_flow_steps, 2] float32
        tensor.
    """
    assert_shapes = tf.debugging.assert_shapes

    # Scatter points across topdown maps for each timestep.  The tensor
    # `point_indices` holds the indices where `should_render_point` is True.
    # It is a 2-D tensor with shape [n, 4], where n is the number of valid
    # agent points inside FOV.  Each row in this tensor contains indices over
    # the following 4 dimensions: (batch, agent, timestep, point).

    # [num_points_to_render, 4]
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)
    # [num_points_to_render, 1]
    x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]
    assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                    (y_img_coord, [num_points_to_render, 1])])

    # [num_points_to_render, 4]
    xy_img_coord = tf.concat(
        [
            point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
            point_indices[:, 2:3],
        ],
        axis=1,
    )
    # [num_points_to_render]
    gt_values_dx = tf.gather_nd(dx, point_indices)
    gt_values_dy = tf.gather_nd(dy, point_indices)

    # tf.scatter_nd() accumulates values when there are repeated indices.
    # Keep track of number of indices writing to the same pixel so we can
    # account for accumulated values.
    # [num_points_to_render]
    gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

    # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
    flow_x = tf.scatter_nd(xy_img_coord, gt_values_dx, topdown_shape)
    flow_y = tf.scatter_nd(xy_img_coord, gt_values_dy, topdown_shape)
    num_values_per_pixel = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
    assert_shapes([
        (flow_x, topdown_shape),
        (flow_y, topdown_shape),
        (num_values_per_pixel, topdown_shape),
    ])

    # Undo the accumulation effect of tf.scatter_nd() for repeated indices.
    flow_x = tf.math.divide_no_nan(flow_x, num_values_per_pixel)
    flow_y = tf.math.divide_no_nan(flow_y, num_values_per_pixel)

    # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps, 2]
    flow = tf.stack([flow_x, flow_y], axis=-1)
    assert_shapes([(flow, topdown_shape + [2])])
    return flow




if __name__ == "__main__":
    ConsistencyLoss()



