import tensorflow as tf
from typing import Tuple
import numpy as np

def torch_gather(x, indices, gather_axis):

    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
        else:
            gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped



def min_ade(traj: tf.Tensor, traj_gt: tf.Tensor, masks: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, nb_actors, num_modes, sequence_length, 3]
    :param traj_gt: ground truth trajectory, shape [batch_size, nb_actors, sequence_length, 3]
    :param masks: masks for varying length ground truth, shape [batch_size, nb_actors, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size, nb_actors]
    """
    num_modes = traj.shape[2]

    traj_gt_rpt = tf.tile(tf.expand_dims(traj_gt, axis=2), [1, 1, num_modes, 1, 1])
    masks_rpt = tf.tile(tf.expand_dims(masks, axis=2), [1, 1, num_modes, 1])
    err = traj_gt_rpt - traj
    err = tf.pow(err, 2)
    err = tf.reduce_sum(err, axis=4)
    err = tf.pow(err, 0.5)
    traj_length = tf.reduce_sum(masks_rpt, axis=3)
    sums = tf.reduce_sum(err * masks_rpt, axis=3)
    err = tf.math.divide_no_nan(sums, traj_length)
    # err = tf.reduce_mean(err, axis=3)
    inds = tf.argmin(err, axis=2)
    err = tf.reduce_min(err, axis=2)

    return err, inds



class TrajLoss(tf.keras.losses.Loss):
    def __init__(self, replica):
        super().__init__()
        self.alpha = 1
        self.beta = 1
        self.replica = replica

        a = tf.concat([tf.random.uniform([4, 48, 6, 8, 3]), tf.zeros([4,16,6,8,3])], axis=1)

        b = tf.concat([tf.random.uniform([4, 48, 6, 8, 3]), tf.ones([4,16,6,8,3])], axis=1)
        probs = tf.concat([tf.ones([4, 64, 1]) * -0.02, tf.ones([4,64,5]) * -2], axis=2)

        dummy_preds = a, probs

        dummy_gt = a[:,:,0,:,:]
        masks = tf.concat([tf.ones([4, 16, 8]), tf.zeros([4,48,8])], axis=1)
        loss = self(dummy_preds, dummy_gt, masks)
        # print(loss)

    def __call__(self, traj_preds_and_probs, gt_trajs, gt_masks):
        # loss : sum of minimum errors across batches and agents

        """
        Compute MTP loss
        :param traj_preds_and_probs (B, 64, 8, 3), (B, 64, 6)
        :param gt_masks for variable length ground truth trajectories (B, 64, 8)
        """

        # Unpack arguments
        trajs, log_probs = traj_preds_and_probs # (B, 64, 6, 8, 3), (B, 64, 6)

        # Obtain mode with minimum ADE with respect to ground truth:
        errs, inds = min_ade(trajs, gt_trajs, gt_masks)

        l_reg = errs

        # create mask of valid trajectories (traj length > 0)
        trajs_length = tf.reduce_sum(gt_masks, axis=2) # (B, 64)
        valid_mask = tf.greater(trajs_length, 0)

        # Compute classification loss
        l_class = - tf.squeeze(torch_gather(log_probs, tf.expand_dims(inds, 2), 2))

        loss = self.beta * l_reg + self.alpha * l_class # (B, 64)

        loss = tf.reduce_mean(tf.boolean_mask(loss, valid_mask)) # (1)

        return loss / self.replica


if __name__ == "__main__":
    trajLoss = TrajLoss()