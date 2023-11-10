import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)
        self.is_train = self.opt.mode == "train"

        # load images path 
        filename = self.opt.train_csv if self.is_train else self.opt.test_csv
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)
        self.imgs_path = self.make_dataset()

        # load AUs dicitionary 
        aus_pkl = os.path.join(self.opt.data_root, self.opt.aus_pkl)
        self.aus_dict = self.load_dict(aus_pkl)

        # load image to tensor transformer
        self.img2tensor = self.img_transformer()

    def make_dataset(self):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), f"Cannot find image file: {img_path}"
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def get_aus_by_path(self, img_path):
        return None

    def img_transformer(self):
        transform_list = []
        if self.opt.resize_or_crop == 'resize_and_crop':
            transform_list.extend(
                (
                    transforms.Resize(
                        [self.opt.load_size, self.opt.load_size], Image.BICUBIC
                    ),
                    transforms.RandomCrop(self.opt.final_size),
                )
            )
        elif self.opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(self.opt.final_size))
        elif self.opt.resize_or_crop == 'none':
            transform_list.append(transforms.Lambda(lambda image: image))
        else:
            raise ValueError(
                f"--resize_or_crop {self.opt.resize_or_crop} is not a valid option."
            )

        if self.is_train and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.extend(
            (
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
        )
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.imgs_path)





    







