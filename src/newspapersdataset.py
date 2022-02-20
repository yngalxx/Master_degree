from PIL import Image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
import pathlib
sys.path.append("/".join(str(pathlib.Path(__file__).parent.resolve()).split('/')[:-2]))
from image_size import get_image_size  # source: https://github.com/scardine/image_size


class NewspapersDataset(Dataset):
    def __init__(self, img_dir, in_list, expected_list, bbox_format='x0y0x1y1', scale=None, transforms=None):
        # selfs
        self.img_dir = img_dir
        self.transforms = transforms
        self.in_list = in_list
        self.expected_list = expected_list
        self.scale = scale

        # read gonito format files and rescale annotations (optional)
        self.df = pd.DataFrame()
        for i in range(len(self.in_list)):
            img_width, img_height = get_image_size.get_image_size(
                self.img_dir + self.in_list[i]
            )
            expected_list_split = self.expected_list[i].split(' ')
            for ii in range(len(expected_list_split)):
                expected_list_split_2 = expected_list_split[ii].split(':')
                bbox = expected_list_split_2[1].split(',')
                if isinstance(self.scale, list):
                    new_img_width, new_img_height = self.scale[0], self.scale[1]
                elif isinstance(self.scale, int) or isinstance(self.scale, float):
                    new_img_width, new_img_height = img_width * self.scale, img_height * self.scale
                else:
                    new_img_width, new_img_height = img_width, img_height
                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                if bbox_format == 'x0y0wh':
                    x1 += x0,
                    y1 += y0
                temp_dict = {
                    'file_name': self.in_list[i],
                    'class': expected_list_split_2[0],
                    'x0': int(x0 / (img_width / new_img_width)),
                    'y0': int(y0 / (img_height / new_img_height)),
                    'x1': int(x1 / (img_width / new_img_width)),
                    'y1': int(y1 / (img_height / new_img_height)),
                    'new_width': int(new_img_width),
                    'new_height': int(new_img_height)
                }
                self.df = self.df.append(temp_dict, ignore_index=True)

    def __getitem__(self, index):
        # read images
        img_name = self.df.file_name[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        temp_df = self.df[self.df.file_name == img_name]
        temp_df = temp_df.reset_index(drop=True)

        # resize
        if self.scale:
            img = img.resize((int(temp_df['new_width'][0]), int(temp_df['new_height'][0])))

        # get bboxes and labels
        boxes, labels = [], []
        for i in range(len(temp_df)):
            x0, y0 = temp_df['x0'][i], temp_df['y0'][i]
            x1, y1 = temp_df['x1'][i], temp_df['y1'][i]
            boxes.append([x0, y0, x1, y1])
            labels.append(int(temp_df['class'][i]))

        # transformation to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([int(img_name.split('.')[0])])

        # wrapping
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id
        }

        # data transformation
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.in_list)
