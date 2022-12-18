
import os 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import matplotlib.pyplot as plt  

import numpy as np
import math

import tensorflow as tf

from typing import List, Mapping, Sequence, Tuple,Dict

from waymo_open_dataset.utils.occupancy_flow_renderer import _sample_and_filter_agent_points,rotate_points_around_origin,_stack_field

from grid_utils import _transform_to_image_coordinates, _rotate_box, add_sdc_fields

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

from tqdm import tqdm
from PIL import Image as Image
from time import time as time

from data_utils import road_label,road_line_map,light_label,light_state_map
import matplotlib as mpl
import argparse
import os
mpl.use('Agg')

feature_desc = {
    'centerlines': tf.io.FixedLenFeature([], tf.string),
    'actors': tf.io.FixedLenFeature([], tf.string),
    'occl_actors': tf.io.FixedLenFeature([], tf.string),
    'ogm': tf.io.FixedLenFeature([], tf.string),
    'map_image': tf.io.FixedLenFeature([], tf.string),
    'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
    'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
    'gt_flow': tf.io.FixedLenFeature([], tf.string),
    'origin_flow': tf.io.FixedLenFeature([], tf.string),
    'vec_flow':tf.io.FixedLenFeature([], tf.string),
    'byc_flow':tf.io.FixedLenFeature([], tf.string)
}

def rotate_all_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> tf.Tensor:

    # FOR TRAJECOTRIES #
    times = ['past','current', 'future']
    x = _stack_field(inputs, times, 'x')
    y = _stack_field(inputs, times, 'y')
    z = _stack_field(inputs, times, 'z')

    vx = _stack_field(inputs, times, 'velocity_x')
    vy = _stack_field(inputs, times, 'velocity_y')

    bbox_yaw = _stack_field(inputs, times, 'bbox_yaw')

    length = _stack_field(inputs, times, 'length')
    width = _stack_field(inputs, times, 'width')

    valid = _stack_field(inputs, times, 'valid')
    valid_indices = tf.cast(tf.equal(valid, 1),tf.float32)
    
    shape = ['batch_size', 'num_agents', 'num_steps', 1]
    tf.debugging.assert_shapes([
        (x, shape),
        (y, shape),
        (vx, shape),
        (vy, shape),
        (z, shape),
        (bbox_yaw, shape)
    ])

    # Translate all agent coordinates such that the autonomous vehicle is at the
    # origin.
    sdc_x = inputs['sdc/current/x'][:, tf.newaxis, tf.newaxis, :]
    sdc_y = inputs['sdc/current/y'][:, tf.newaxis, tf.newaxis, :]
    sdc_z = inputs['sdc/current/z'][:, tf.newaxis, tf.newaxis, :]

    x = x - sdc_x
    y = y - sdc_y
    z = z - sdc_z

    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw'][:, tf.newaxis,
                                                            tf.newaxis, :]
    
    x, y = rotate_points_around_origin(x, y, angle)

    _,_,psudo_occu_mask = _transform_to_image_coordinates(x[:,:,10,:], y[:,:,10,:], config,larger_box=True) # at current timestep

    ul_x,ul_y, ur_x,ur_y, ll_x,ll_y, lr_x,lr_y = _rotate_box(x,y,length,width,bbox_yaw+angle)

    _,_,in_box_lu = _transform_to_image_coordinates(ul_x,ul_y,config)
    _,_,in_box_ru = _transform_to_image_coordinates(ur_x,ur_y,config)
    _,_,in_box_ld = _transform_to_image_coordinates(ll_x,ll_y,config)
    _,_,in_box_rd = _transform_to_image_coordinates(lr_x,lr_y,config)

    in_box = tf.logical_or(
        tf.logical_or(in_box_lu,in_box_ru), 
        tf.logical_or(in_box_ld,in_box_rd)
        )

    # print(in_box.get_shape())
    in_box_mask = tf.not_equal(tf.reduce_sum(tf.cast(in_box,tf.int32)[:,:,:11,0],axis=-1),0) # check if seen in past or present (11 steps)

    occu_mask = tf.logical_and(psudo_occu_mask[:,:,0],tf.logical_not(in_box_mask))
    # print(in_box_mask.get_shape())
    # vx, vy = rotate_points_around_origin(vx, vy, angle)
    bbox_yaw = bbox_yaw + angle

    actor_traj = tf.multiply(valid_indices,tf.concat([x,y,bbox_yaw], axis=-1))

    info = tf.stack([length, width], axis=3)

    return actor_traj,in_box_mask,occu_mask,valid, info

def extract_lines(xy, id, typ):
    line = [] # a list of points  
    lines = [] # a list of lines
    length = xy.shape[0]
    for i, p in enumerate(xy):
        line.append(p)
        next_id = id[i+1] if i < length-1 else id[i]
        current_id = id[i]
        if next_id != current_id or i == length-1:
            if typ in [18, 19]:
                line.append(line[0])
            lines.append(line)
            line = []
    return lines

class Processor(object):

    def __init__(self, area_size, max_actors,max_occu, radius,rasterisation_size=256,save_dir='.',ids_dir=''):
        # parameters
        self.img_size = rasterisation_size # size = pixels * pixels
        self.area_size = area_size # size = [vehicle, pedestrian, cyclist] meters * meters
        self.max_actors = max_actors
        self.max_occu = max_occu
        self.radius = radius
        self.save_dir = save_dir
        self.ids_dir = ids_dir

        self.get_config()

    def load_data(self, filename):
        self.filename = filename
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        self.dataset_length = len(list(dataset.as_numpy_iterator()))
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        self.datalist = dataset.batch(1)
        # self.datalist = list(dataset.as_numpy_iterator())
    
    def get_config(self):
        config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        config_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 256
        grid_width_cells: 256
        sdc_y_in_grid: 192
        sdc_x_in_grid: 128
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(config_text, config)

        self.config = config

        ogm_config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        oconfig_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 512
        grid_width_cells: 512
        sdc_y_in_grid: 320
        sdc_x_in_grid: 256
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(oconfig_text, ogm_config)
        self.ogm_config = ogm_config

    def read_data(self, parsed):
        
        actor_traj,traj_mask,occu_mask,actor_valid, info = rotate_all_from_inputs(parsed, self.config)
        
        # actor traj
        self.actor_traj = actor_traj[0].numpy()
        self.traj_mask = traj_mask[0,:].numpy()
        self.occu_mask = occu_mask[0,:].numpy()
        #[batch,actor_num,91,1]
        self.actor_valid = actor_valid[0,:,:,0].numpy()
        self.actor_type = parsed['state/type'][0].numpy()

        # get type, length and width
        self.info = info[0, :, :, :].numpy() # (128, 91, 3)


    def actor_traj_process(self):
        emb = np.eye(3)
        traj_m = np.where(self.traj_mask)
        valid_actor = self.actor_traj[traj_m]
        valid_mask = self.actor_valid[traj_m]
        valid_type = self.actor_type[traj_m] # (found)
        valid_info = self.info[traj_m] # (found, 91, 2)
        dist=[]
        curr_buf=[]

        for i in range(valid_actor.shape[0]):
            w = np.where(valid_mask[i])[0]
            if w.shape[0]==0:
                continue
            n = min(10,w[-1])
            last_pos = valid_actor[i,n,:]
            dist.append(last_pos[:2])

        dist = np.argsort(np.linalg.norm(dist,axis=-1))[:self.max_actors]

        output_actors = np.zeros((self.max_actors,91,3))
        output_actors_info = np.zeros((self.max_actors,3))
        output_actors_ref = np.zeros((self.max_actors, 4))
        output_actors_mask = np.zeros((self.max_actors,91))
        for i,d in enumerate(dist):

            # transform to local
            w = np.where(valid_mask[d])[0]
            n = min(10, w[-1]) # at most the current timestep
            last_known = valid_actor[d,n,:] # last known <= current timestep

            ref_x, ref_y, ref_yaw = last_known

            # save ref
            output_actors_ref[i, :] = [*last_known, valid_mask[d, n]]

            x = valid_actor[d, :, 0]
            y = valid_actor[d, :, 1]
            yaw = valid_actor[d, :, 2]


            x = x - ref_x # (91)
            y = y - ref_y # (91)

            angle = math.pi / 2 - ref_yaw

            tx = np.cos(angle) * x - np.sin(angle) * y
            ty = np.sin(angle) * x + np.cos(angle) * y
            x, y = tx, ty

            yaw = yaw + angle

            actor = np.stack([x,y,yaw], axis=1) # (91, 3)

            m = w[-1]+1 # end of trajectory (from last valid + 1)
            actor[:, m:] = 0 # set to 0 end of trajectory

            output_actors[i] = actor
            output_actors_info[i] = [valid_type[d], *valid_info[d, n, :]]
            output_actors_mask[i] = valid_mask[d]
        
        #process the possible occulde traj:
        occ_m = np.where(self.occu_mask)
        occu_actor = self.actor_traj[occ_m]
        occu_valid = self.actor_valid[occ_m]
        occu_type = self.actor_type[occ_m]
        occu_info = self.info[occ_m] # (found, 91, 2)

        dist=[]
        curr_buf=[]
        occu_traj = []
        o_type = []
        for i in range(occu_actor.shape[0]):
            w = np.where(occu_valid[i])[0]
            if w.shape[0]==0:
                continue
            b,e = w[0] , min(10, w[-1])
            begin_pos,last_pos = occu_actor[i,b,:2],occu_actor[i,e,:2]
            begin_dist,last_dist = np.linalg.norm(begin_pos) , np.linalg.norm(last_pos)
            if begin_dist<=last_dist:
                continue
            dist.append(last_dist)
            # curr_buf.append(occu_actor[i,e,:])
            occu_traj.append(occu_actor[i])
            o_type.append(occu_type[i])
        
        dist = np.argsort(dist)[:self.max_occu]

        output_occu_actors = np.zeros((self.max_occu,91,3))
        output_occu_actors_info = np.zeros((self.max_occu, 3))
        output_occu_actors_ref = np.zeros((self.max_occu, 4))
        output_occu_actors_mask = np.zeros((self.max_occu, 91))
        for i,d in enumerate(dist):

            # transform to local
            w = np.where(occu_valid[d])[0]
            n = min(10, w[-1]) # at most the current timestep
            last_known = occu_traj[d][n,:] # last known <= current timestep

            ref_x, ref_y, ref_yaw = last_known

            # save ref
            output_occu_actors_ref[i, :] = [*last_known, occu_valid[d, n]]

            x = occu_traj[d][:, 0]
            y = occu_traj[d][:, 1]
            yaw = occu_traj[d][:, 2]

            x = x - ref_x # (91)
            y = y - ref_y # (91)

            angle = math.pi / 2 - ref_yaw

            tx = np.cos(angle) * x - np.sin(angle) * y
            ty = np.sin(angle) * x + np.cos(angle) * y
            x, y = tx, ty

            yaw = yaw + angle

            occu_actor = np.stack([x,y,yaw], axis=1) # (91, 3)

            m = w[-1]+1 # end of trajectory (from last valid + 1)
            occu_actor[:, m:] = 0 # set to 0 end of trajectory

            output_occu_actors[i] = occu_actor 
            output_occu_actors_info[i] = [o_type[d], *occu_info[d, n, :]]
            output_occu_actors_mask[i] = occu_valid[d]

        # combine obs and occl. actors
        output_all_trajs = np.concatenate([output_actors, output_occu_actors], axis=0) # (64, 91, 3) (x, y, yaw)
        output_all_infos = np.concatenate([output_actors_info, output_occu_actors_info], axis=0) # (64, 2) (length, width)
        output_all_masks = np.concatenate([output_actors_mask, output_occu_actors_mask], axis=0) # (64, 91)
        output_all_refs = np.concatenate([output_actors_ref, output_occu_actors_ref], axis=0) # (64, 91, 4)

        return output_all_trajs , output_all_infos, output_all_masks, output_all_refs
    
    def get_ids(self,val=True):
        if val:
            path = f'{self.ids_dir}/validation_scenario_ids.txt'
        else:
            path = f'{self.ids_dir}/testing_scenario_ids.txt'
        with tf.io.gfile.GFile(path) as f:
            test_scenario_ids = f.readlines()
            test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
            self.test_scenario_ids = set(test_scenario_ids)
    
    def build_saving_tfrecords(self,pred,val,num):
        if val:
            self.get_ids(val=True)
            if not os.path.exists(f'{self.save_dir}/val_trajs/'):
                os.makedirs(f'{self.save_dir}/val_trajs/')
            writer = tf.io.TFRecordWriter(f'{self.save_dir}/val_trajs/'+f'{num}'+'new3.tfrecords', options="GZIP")
        
        if not (pred or val):
            if not os.path.exists(f'{self.save_dir}/train_trajs/'):
                os.makedirs(f'{self.save_dir}/train_trajs/')
            writer = tf.io.TFRecordWriter(f'{self.save_dir}/train_trajs/'+f'{num}'+'new3.tfrecords', options="GZIP")
        return writer

    def get_already_processed(self, pred, val, num):
        if val:
            processed = tf.data.TFRecordDataset(f'{self.save_dir}/val/'+f'{num}'+'new.tfrecords', compression_type="GZIP")

        if not (pred or val):
            processed = tf.data.TFRecordDataset(f'{self.save_dir}/train/'+f'{num}'+'new.tfrecords', compression_type="GZIP")
        return processed
        
    def workflow(self,pred=False,val=False):

        if (pred):
            return print("Can't get ground truth trajectories for the test dataset!")


        i = 0
        self.pbar = tqdm(total=self.dataset_length)
        num = self.filename.split('-')[1]
        writer = self.build_saving_tfrecords(pred, val,num)
        already_processed = self.get_already_processed(pred, val, num)
        
        for dataframe, raw_feature in zip(self.datalist, already_processed):
            feature =  tf.io.parse_single_example(raw_feature, feature_desc)
            if pred or val:
                sc_id = dataframe['scenario/id'].numpy()[0]
                if isinstance(sc_id, bytes):
                    sc_id=str(sc_id, encoding = "utf-8") 
                if sc_id not in self.test_scenario_ids:
                    self.pbar.update(1)
                    continue

            dataframe = add_sdc_fields(dataframe)
            self.read_data(dataframe)

            output_all_trajs, output_all_infos, output_all_masks, output_all_refs = self.actor_traj_process()

            gt_trajs = output_all_trajs[:, 20::10, :] # (64, 8, 3)
            gt_infos = output_all_infos # (64, 3) (type, length, width)
            gt_masks = output_all_masks[:, 20::10] # (64, 8)
            gt_refs = output_all_refs # (64, 4) (x, y, yaw, valid)

            new_feature = {
                'gt_trajs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_trajs.tobytes()])),
                'gt_infos': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_infos.tobytes()])),
                'gt_masks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_masks.tobytes()])),
                'gt_refs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_refs.tobytes()]))
            }

            for k,v in feature.items():
                new_feature[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.numpy()]))
            

            example = tf.train.Example(features=tf.train.Features(feature=new_feature))
            writer.write(example.SerializeToString())
            self.pbar.update(1)
            i+=1
            # if i>=64:
            #     break

        writer.close()
        self.pbar.close()
        print('collect:',i)


def process_training_data(filename):
    print('Working on',filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow()
    print(filename, 'done!')

def process_val_data(filename):
    print('Working on', filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow(val=True)
    print(filename, 'done!')

if __name__=="__main__":
    from multiprocessing import Pool
    from glob import glob

    parser = argparse.ArgumentParser(description='Data-preprocessing')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="/datasets/waymo110/occupancy_flow_challenge")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="/datasets/waymo110/preprocessed_data")
    parser.add_argument('--file_dir', type=str, help='Dataset directory',default="/datasets/waymo110/tf_example")
    parser.add_argument('--pool', type=int, help='num of pooling multi-processes in preprocessing',default=1)
    args = parser.parse_args()

    NUM_POOLS = args.pool

    train_files = glob(f'{args.file_dir}/training/*')
    print(f'Processing training data...{len(train_files)} found!')
    print('Starting processing pooling...')
    with Pool(NUM_POOLS) as p:
        p.map(process_training_data, train_files)
    
    val_files = glob(f'{args.file_dir}/validation/*')
    print(f'Processing validation data...{len(val_files)} found!')
    print('Starting processing pooling...')
    with Pool(NUM_POOLS) as p:
        p.map(process_val_data, val_files)




    