import os
import numpy as np
from torch.utils import data
from .semantickitti_utils import LEARNING_MAP
# from .LaserMix_semantickitti import lasermix_aug
# from .PolarMix_semantickitti import polarmix
# import random



def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SemantickittiDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training: bool = True,
        class_names: list = None,
        root_path: str = None,
        logger = None,
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
        self.if_scribble = if_scribble
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

        if self.split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'train_val':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise Exception('split must be train/val/train_val/test.')
        
        self.annos = []
        if self.split == 'train':
            for seq in self.seqs:
                self.annos += absoluteFilePaths('/'.join([self.root_path, str(seq).zfill(2), 'velodyne']))
        elif self.split == 'val': # corruption evaluation is True
            self.annos += absoluteFilePaths('/'.join([self.root_path, 'velodyne']))



        print(f'The total sample is {len(self.annos)}')
        self._sample_idx = np.arange(len(self.annos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.annos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)
    

    def __getitem__(self, index):
        raw_data = np.fromfile(self.annos[index], dtype=np.float32).reshape((-1, 4))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.if_scribble:  # ScribbleKITTI (weak label)
                annos = self.annos[index].replace('SemanticKITTI', 'ScribbleKITTI')
                annotated_data = np.fromfile(
                    annos.replace('velodyne', 'scribbles')[:-3] + 'label', dtype=np.uint32
                ).reshape((-1, 1))
            else:  # SemanticKITTI (full label)
                annotated_data = np.fromfile(
                    self.annos[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32
                ).reshape((-1, 1))
            
            annotated_data = annotated_data & 0xFFFF
            annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data)

        
        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': self.annos[index],
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

