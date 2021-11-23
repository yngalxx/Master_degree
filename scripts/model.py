from PIL import Image
import pandas as pd
import csv
import os
import sys
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN

sys.path.append('../../')
from image_size import get_image_size  # source: https://github.com/scardine/image_size

DIR_PATH = '/Users/alexdrozdz/Desktop/Studia/00. Seminarium magisterskie/'


def from_tsv_to_list(path):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=";")

    expected = list(read_tsv)[0]

    return expected


# train
expected_train = from_tsv_to_list(DIR_PATH + 'Master_gonito/train/expected.tsv')
in_train = from_tsv_to_list(DIR_PATH + 'Master_gonito/train/in.tsv')

# val
expected_val = from_tsv_to_list(DIR_PATH + 'Master_gonito/dev-0/expected.tsv')
in_val = from_tsv_to_list(DIR_PATH + 'Master_gonito/dev-0/in.tsv')

# test
expected_test = from_tsv_to_list(DIR_PATH + 'Master_gonito/test-A/expected.tsv')
in_test = from_tsv_to_list(DIR_PATH + 'Master_gonito/test-A/in.tsv')


class NewspapersDataset(Dataset):
    def __init__(self, img_dir, in_list, expected_list, transform=None, scale=None):
        # selfs
        self.img_dir = img_dir
        self.transform = transform
        self.in_list = in_list
        self.expected_list = expected_list
        self.scale = scale

        # read gonito format files and rescale annotations (optional)
        self.df = pd.DataFrame()
        for i in range(len(self.in_list)):
            if self.scale:
                img_width, img_height = get_image_size.get_image_size(
                    img_dir + self.in_list[i]
                )
                if len(self.scale) == 2:
                    self.scaler_w = self.scale[0] / img_width
                    self.scaler_h = self.scale[1] / img_height
                elif len(self.scale) == 1:
                    if img_height > img_width:
                        self.scaler_w = self.scale[0] / img_width
                        self.scaler_h = self.scale[0] / img_width
                    elif img_height < img_width:
                        self.scaler_w = self.scale[0] / img_height
                        self.scaler_h = self.scale[0] / img_height
                    elif img_height == img_width:
                        self.scaler_w = self.scale[0] / img_width
                        self.scaler_h = self.scale[0] / img_height
            else:
                self.scaler_w, self.scaler_h = 1, 1
            expected_list_split = self.expected_list[i].split(' ')
            for ii in range(len(expected_list_split)):
                expected_list_split_2 = expected_list_split[ii].split('/')
                bbox = expected_list_split_2[1].split(',')
                temp_dict = {
                    'file_name': self.in_list[i],
                    'class': expected_list_split_2[0],
                    'x0': int(int(bbox[0]) * self.scaler_w),
                    'y0': int(int(bbox[1]) * self.scaler_h),
                    'x1': int((int(bbox[0]) + int(bbox[2])) * self.scaler_w),
                    'y1': int((int(bbox[1]) + int(bbox[3])) * self.scaler_h),
                }
                self.df = self.df.append(temp_dict, ignore_index=True)

    def __len__(self):
        return len(self.in_list)

    def __getitem__(self, index):
        # read images
        img_name = self.df.file_name[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)

        # get bboxes and labels
        temp_df = self.df[self.df.file_name == img_name]
        boxes, labels = [], []
        temp_df = temp_df.reset_index(drop=True)
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
        image_id = torch.tensor([index])

        # wrapping
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id
        }

        # data transformation (optional)
        if self.transform:
            img = self.transform(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# hyperparameters
IN_CHANNEL = 1
NUM_CLASSES = 8  # 7 classes, but there is also one for background (?)
LEARNING_RATE = 5e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
RESIZE = [800, ]

# load datasets
image_dir = DIR_PATH + 'scraped_photos_final/'
data_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE),
        transforms.Grayscale(num_output_channels=IN_CHANNEL),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# train_dataloader = DataLoader(
#     NewspapersDataset(
#         image_dir,
#         in_train,
#         expected_train,
#         transform=data_transform,
#         scale=RESIZE
#     ),
#     batch_size=BATCH_SIZE,
#     collate_fn=collate_fn
# )

val_dataloader = DataLoader(
    NewspapersDataset(
        image_dir,
        in_val,
        expected_val,
        transform=data_transform,
        scale=RESIZE
    ),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# test_dataloader = DataLoader(
#     NewspapersDataset(
#         image_dir,
#         in_test,
#         expected_test,
#         transform=data_transform,
#         scale=RESIZE
#     ),
#     batch_size=BATCH_SIZE,
#     collate_fn=collate_fn
# )

# pre-trained model
resnet50 = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True
)

resnet50.out_channels = 1280

# freeze layers (gradient will not be calculated during backpropagation)
for param in resnet50.parameters():
    param.requires_grad = False

# we need to change the predictor layers to match the number of classes in our custom dataset
resnet50.roi_heads.box_predictor.cls_score = nn.Linear(
    resnet50.roi_heads.box_head.fc7.out_features,
    out_features=NUM_CLASSES
)
resnet50.roi_heads.box_predictor.bbox_pred = nn.Linear(
    resnet50.roi_heads.box_head.fc7.out_features,
    out_features=NUM_CLASSES * 4
)

# we need to change the predictor layers to match the number of classes in our custom dataset
resnet50.roi_heads.box_predictor.cls_score = nn.Linear(
    resnet50.roi_heads.box_head.fc7.out_features,
    out_features=NUM_CLASSES
)
resnet50.roi_heads.box_predictor.bbox_pred = nn.Linear(
    resnet50.roi_heads.box_head.fc7.out_features,
    out_features=NUM_CLASSES * 4
)

# the optimizer needs to be modified (because of frozen parameters) to only get the parameters with requires_grad=True
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, resnet50.parameters()),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)

# learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# main model
model = FasterRCNN(
    backbone=resnet50,
    num_classes=NUM_CLASSES,
)

# switch to gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

start_time = datetime.datetime.now()
model.train()
for epoch in range(NUM_EPOCHS):
    train_loss = 0.0
    #    for images, targets in tqdm(train_dataloader):
    for images, targets in tqdm(val_dataloader):
        # clear the gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(images, targets)
        # loss
        loss = F.cross_entropy(outputs, targets)
        # calculate gradients
        loss.backward()
        # update weights
        optimizer.step()
        # calculate loss
        train_loss += loss.item()

    #     print(f'{datetime.datetime.now()} - epoch {epoch+1}: loss = {train_loss / len(train_dataloader)}')
    print(f'{datetime.datetime.now()} - epoch {epoch + 1}: loss = {train_loss / len(val_dataloader)}')

    lr_scheduler.step()

print(f'\nModel training completed, runtime: {datetime.datetime.now() - start_time}')
