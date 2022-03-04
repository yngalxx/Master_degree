import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from newspapersdataset import NewspapersDataset
from newspapersdataset import prepare_data_for_dataloader
from train_model import train_model
from test_model import model_predict
from functions import from_tsv_to_list
from functions import collate_fn
from functions import seed_worker
import warnings
import numpy as np

# warnings
warnings.filterwarnings("ignore")

# hyperparameters and more
parameters = {
    'channel': 1, # 3 <= RGB, 1 <= greyscale
    'num_classes': 8, # 7 classes, but there is also one for background
    'learning_rate': 1e-4,
    'batch_size': 16,
    'num_epochs': 8,
    'rescale': [375, 500], # if float, each image will be multiplied by it, if list [width, height] each image will be scaled to that size (concerns both images + annotations)
    'shuffle': False, 
    'weight_decay': 0, # regularization
    'lr_step_size': 1, # lr scheduler step
    'lr_gamma': 0.9, # lr step multiplier 
    'trainable_backbone_layers': 5, # 5 <= all, 0 <= any
    'num_workers': 4,
    'main_dir': '/home/wmi/adrozdz/Master_degree/',
    'image_dir': '/home/wmi/adrozdz/scraped_photos_final/',
    'annotations_dir': '/home/wmi/adrozdz/Master_gonito/',
    'train': True,
    'test': True,
    'val': True,
    'gpu': True,
}

# read data and create dataloaders 
data_transform = T.Compose([
    T.Grayscale(num_output_channels=parameters['channel']),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
    ])

if parameters['val']:
    expected_val = from_tsv_to_list(parameters['annotations_dir']+'dev-0/expected.tsv')
    in_val = from_tsv_to_list(parameters['annotations_dir']+'dev-0/in.tsv')
    val_paths = [parameters['image_dir']+path for path in in_val]
    data_val = prepare_data_for_dataloader(
        img_dir=parameters['image_dir'],
        in_list=in_val,
        expected_list=expected_val,
        bbox_format='x0y0x1y1',
        scale=parameters['rescale'],
        test=False,
        )
    dataset_val = NewspapersDataset(
        df=data_val,
        images_path=val_paths,
        scale=parameters['rescale'],
        transforms=data_transform,
        test=False,
        )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=parameters['batch_size'],
        shuffle=parameters['shuffle'],
        collate_fn=collate_fn,
        num_workers=parameters['num_workers'],
        )
else:
    val_dataloader=None

if parameters['train']:
    expected_train = from_tsv_to_list(parameters['annotations_dir']+'train/expected.tsv')
    in_train = from_tsv_to_list(parameters['annotations_dir']+'train/in.tsv')
    train_paths = [parameters['image_dir']+path for path in in_train]
    data_train = prepare_data_for_dataloader(
        img_dir=parameters['image_dir'],
        in_list=in_train,
        expected_list=expected_train,
        bbox_format='x0y0x1y1',
        scale=parameters['rescale'],
        test=False,
        )
    dataset_train = NewspapersDataset(
        df=data_train,
        images_path=train_paths,
        scale=parameters['rescale'],
        transforms=data_transform,
        test=False,
        )
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=parameters['batch_size'],
        shuffle=parameters['shuffle'],
        collate_fn=collate_fn,
        num_workers=parameters['num_workers'],
    )

    # pre-trained model as a backbone
    resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers=parameters['trainable_backbone_layers']  
    )
    backbone = resnet.backbone

    # main model
    model = FasterRCNN(
        backbone,
        num_classes=parameters['num_classes'],
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
        batch_size_per_image=parameters['batch_size'],
        positive_fraction=0.5,
        pre_nms_top_n=dict(training=200, testing=100),
        post_nms_top_n=dict(training=160, testing=80),
        nms_thresh=0.7
    )

    # optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=parameters['learning_rate'],
        weight_decay=parameters['weight_decay']
    )

    # learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=parameters['lr_step_size'],
        gamma=parameters['lr_gamma']
    )

    # training
    trained_model = train_model(
        model=model, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader,
        epochs=parameters['num_epochs'], 
        gpu=parameters['gpu'],
        save_path=parameters['main_dir'],
        val_dataloader=val_dataloader, 
        lr_scheduler=lr_scheduler, 
    )

# prediction
if parameters['test']:
    model = torch.load(parameters['main_dir']+'saved_models/model.pth')
    
    in_test = from_tsv_to_list(parameters['annotations_dir']+'test-A/in.tsv')
    test_paths = [parameters['image_dir']+path for path in in_test]
    data_test = prepare_data_for_dataloader(
        img_dir=parameters['image_dir'],
        in_list=in_test,
        scale=parameters['rescale'],
        test=True,
        )
    dataset_test = NewspapersDataset(
        df=data_test,
        images_path=test_paths,
        scale=parameters['rescale'],
        transforms=data_transform,
        test=True,
        )
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=parameters['batch_size'],
        shuffle=parameters['shuffle'],
        collate_fn=collate_fn,
        num_workers=parameters['num_workers'],
    )

    # prediction on test set
    model_predict(
        model=model, 
        test_dataloader=test_dataloader,
        gpu=parameters['gpu'],
        save_path=parameters['main_dir']+'model_output/test_model_output.csv',
    )
    # prediction on train set (to check under/overfitting)
    if parameters['train']:
        model_predict(
            model=model, 
            test_dataloader=train_dataloader,
            gpu=parameters['gpu'],
            save_path=parameters['main_dir']+'model_output/train_model_output.csv',
        )
    # prediction on validation set
    if parameters['val']:
        model_predict(
            model=model, 
            test_dataloader=val_dataloader,
            gpu=parameters['gpu'],
            save_path=parameters['main_dir']+'model_output/val_model_output.csv',
        )
