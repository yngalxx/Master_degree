import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from train_model import train_model
from functions import from_tsv_to_list, collate_fn
# from train_model import NewspaperNeuralNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from newspapersdataset import NewspapersDataset
import warnings
warnings.filterwarnings("ignore")

# warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

# hyperparameters
# parameters = {
#     'channel': 1, # 3 <= RGB, 1 <= greyscale
#     'num_classes': 8, # 7 classes, but there is also one for background
#     'learning_rate': 1e-4,
#     'batch_size': 16,
#     'num_epochs': 5,
#     'rescale': [450, 600], # if float, each image will be multiplied by it, if list [width, height] each image will be scaled to that size (concerns both images + annotations)
#     'shuffle': True, 
#     'weight_decay': 0, # regularization
#     'step_size': 2, # lr scheduler step
#     'gamma': 0.1, # lr step multiplier 
#     'trainable_backbone_layers': 5, # 5 <= all, 0 <= any
#     'num_workers': 4,
#     'main_dir': '/home/wmi/adrozdz/Master_degree/',
#     'image_dir': '/home/wmi/adrozdz/scraped_photos_final/',
#     'annotations_dir': '/home/wmi/adrozdz/Master_gonito/'
# }

# directories
MAIN_DIR = '/home/wmi/adrozdz/Master_degree/'
DIR_IMAGE = '/home/wmi/adrozdz/scraped_photos_final/'
GONITO_DIR = '/home/wmi/adrozdz/Master_gonito/'

# hyperparameters
CHANNEL = 1 # 3 <= RGB, 1 <= greyscale
NUM_CLASSES = 8  # 7 classes, but there is also one for background
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
NUM_EPOCHS = 5
RESCALE = [375, 500]  # if float, each image will be multiplied by it, if list [width, height] each image will be scaled
# to that size (concerns both images + annotations)
SHUFFLE = True
TRAINABLE_BACKBONE_LAYERS = 5 # 5 <= all, 0 <= any
WEIGHT_DECAY = 0 # regularization
STEP_SIZE = 2 # lr scheduler step
GAMMA = 0.1 # lr step multiplier 
NUM_WORKERS = 4

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
    collate_fn=collate_fn,
    num_workers = NUM_WORKERS
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
    trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS#parameters['trainable_backbone_layers']  
)
backbone = resnet.backbone

# main model
model = FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES#parameters['num_classes'],
)

# # module that generates the anchors for a set of feature maps
# anchor_generator = AnchorGenerator(
#     sizes=tuple([(16, 32, 64, 128, 256) for _ in range(5)]),
#     aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)])
# )

# # module that computes the objectness and regression deltas from the RPN
# rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

# # region proposal network
# model.rpn = RegionProposalNetwork(
#     anchor_generator=anchor_generator,
#     head=rpn_head,
#     fg_iou_thresh=0.7,
#     bg_iou_thresh=0.3,
#     batch_size_per_image=BATCH_SIZE,#parameters['batch_size'],
#     positive_fraction=0.5,
#     pre_nms_top_n=dict(training=200, testing=100),
#     post_nms_top_n=dict(training=160, testing=80),
#     nms_thresh=0.7
# )

# optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=STEP_SIZE,
    gamma=GAMMA
)

trained_model = train_model(
    model, 
    optimizer, 
    train_dataloader,
    NUM_EPOCHS, 
    gpu = True,
    save_path = MAIN_DIR,
    val_dataloader=val_dataloader, 
    test_dataloader=test_dataloader, 
    lr_scheduler=None#lr_scheduler, 
)

# # model instance
# nnn_model = NewspaperNeuralNet(model=model, parameters=parameters)

# # loger
# tb_logger = TensorBoardLogger(save_dir=parameters['main_dir']+'logs/')

# trainer = pl.Trainer(
#     logger=[tb_logger], 
#     gpus=1, 
#     auto_select_gpus=True
# )

# trainer.fit(nnn_model)