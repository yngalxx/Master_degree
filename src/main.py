import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from newspapersdataset import NewspapersDataset
from train_model import train_model
from functions import from_tsv_to_list, collate_fn

# directories
DIR_IMAGE = 'scraped_photos_final/'
GONITO_DIR = 'gonito_files/'

# hyperparameters
CHANNEL = 1
NUM_CLASSES = 8  # 7 classes, but there is also one for background
LEARNING_RATE = 5e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
RESCALE = 1 / 6  # if float, each image will be multiplied by it, if list [width, height] each image will be scaled
# to that size (concerns both images + annotations)
SHUFFLE = True

# read data and create dataloaders
data_transform = T.Compose([
    T.Grayscale(num_output_channels=CHANNEL),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])
## train
expected_train = from_tsv_to_list(GONITO_DIR + 'train/expected.tsv')
in_train = from_tsv_to_list(GONITO_DIR + 'train/in.tsv')
train_dataloader = DataLoader(
    NewspapersDataset(
        img_dir=DIR_IMAGE,
        in_list=in_train,
        expected_list=expected_train,
        scale=RESCALE,
        transforms=data_transform
    ),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    collate_fn=collate_fn
)
## val
expected_val = from_tsv_to_list(GONITO_DIR + 'dev-0/expected.tsv')
in_val = from_tsv_to_list(GONITO_DIR + 'dev-0/in.tsv')
val_dataloader = DataLoader(
    NewspapersDataset(
        img_dir=DIR_IMAGE,
        in_list=in_val,
        expected_list=expected_val,
        scale=RESCALE,
        transforms=data_transform
    ),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    collate_fn=collate_fn
)
## test
expected_test = from_tsv_to_list(GONITO_DIR + 'test-A/expected.tsv')
in_test = from_tsv_to_list(GONITO_DIR + 'test-A/in.tsv')
test_dataloader = DataLoader(
    NewspapersDataset(
        img_dir=DIR_IMAGE,
        in_list=in_test,
        expected_list=expected_test,
        scale=RESCALE,
        transforms=data_transform
    ),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    collate_fn=collate_fn
)

# pre-trained model as a backbone
resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    trainable_backbone_layers=0  # 5 <= all, 0 <= any
)
backbone = resnet.backbone

# main model
model = FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES,
)

# module that generates the anchors for a set of feature maps
anchor_generator = AnchorGenerator(
    sizes=tuple([(16, 32, 64, 128, 256) for _ in range(5)]),
    aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)])
)

# module that computes the objectness and regression deltas from the RPN
rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

# region proposal network
model.rpn = RegionProposalNetwork(
    anchor_generator=anchor_generator,
    head=rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=BATCH_SIZE,
    positive_fraction=0.5,
    pre_nms_top_n=dict(training=200, testing=100),
    post_nms_top_n=dict(training=160, testing=80),
    nms_thresh=0.7
)

# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)

# learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

trained_model = train_model(
    model, optimizer, train_dataloader, NUM_EPOCHS, val_dataloader=val_dataloader, lr_scheduler=lr_scheduler
)
