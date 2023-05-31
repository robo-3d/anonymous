import os
import numpy as np
from torch.utils import data
import random
import pickle
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes
from .nuscenes_utils import LEARNING_MAP

class NuscDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training: bool = True,
        class_names: list = None,
        root_path: str = None,
        logger = None,
        nusc=None,
        if_scribble: bool = False,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.root_path = root_path
        self.training = training
        self.logger = logger
        self.class_names = class_names
        self.tta = data_cfgs.get('TTA', False)
        self.train_val = data_cfgs.get('TRAINVAL', False)
        self.augment = data_cfgs.AUGMENT
        self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.root_path, verbose=False
            )
        if_corrupt = data_cfgs.get('CORRUPT', False)
        if if_corrupt:
            self.root_path = data_cfgs.get('CORRUPT_ROOT')
        if self.training and not self.train_val:
            self.split = 'train'
        else:
            if self.training and self.train_val:
                self.split = 'train_val'
            else:
                self.split = 'val'
        if self.tta:
            self.split = 'test'

        self.annos = []

        if self.split in ("train", "val", "test"):
             phase_scenes = create_splits_scenes()[self.split]
        skip_counter = 0
        skip_ratio = 1  
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_tokens(scene)
        
        print(f'The total sample is {len(self.annos)}')


    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            self.annos.append(current_sample["data"]["LIDAR_TOP"])
            current_sample_token = next_sample_token

    def __len__(self):
        return len(self.annos)


    def __getitem__(self, index):
        lidar_token = self.annos[index]
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.root_path, pointsensor["filename"])

        raw_data = LidarPointCloud.from_file(pcl_path).points.T
        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(
                self.root_path, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
            annotated_data = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1,1)
            annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data)
            
        pc_data = {
            'xyzret': raw_data.astype(np.float32),
            'labels': annotated_data.astype(np.uint8),
            'path': pcl_path,
        }

        return pc_data
    
    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

