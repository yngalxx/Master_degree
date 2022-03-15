import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
    'learning_rate': 3e-4,
    'batch_size': 16,
    'num_epochs': 20,
    'rescale': [1000, 1000], # if float, each image will be multiplied by it, if list [width, height] each image will be scaled to that size (concerns both images + annotations)
    'shuffle': False, 
    'weight_decay': 0, # regularization
    'lr_scheduler': True, # lr scheduler
    'lr_step_size': 5, # lr scheduler step
    'lr_gamma': .4, # lr step multiplier 
    'trainable_backbone_layers': 5, # 5 <= all, 0 <= any
    'num_workers': 2,
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

    # pre-trained resnet50 model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers=parameters['trainable_backbone_layers']
    )

    # replace the pre-trained head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 
        num_classes=parameters['num_classes']
        )

    # optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=parameters['learning_rate'],
        weight_decay=parameters['weight_decay']
    )
    # learning rate scheduler decreases the learning rate by 'gamma' every 'step_size'
    if parameters['lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=parameters['lr_step_size'],
            gamma=parameters['lr_gamma']
        )
    else:
        lr_scheduler = None

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
    print('###  Evaluating test set  ###')
    model_predict(
        model=model, 
        test_dataloader=test_dataloader,
        gpu=parameters['gpu'],
        save_path=parameters['main_dir']+'model_output/test_model_output.csv',
    )
    # prediction on train set (to check under/overfitting)
    if parameters['train']:
        print('###  Evaluating train set  ###')
        model_predict(
            model=model, 
            test_dataloader=train_dataloader,
            gpu=parameters['gpu'],
            save_path=parameters['main_dir']+'model_output/train_model_output.csv',
        )
    # prediction on validation set
    if parameters['val']:
        print('###  Evaluating validation set  ###')
        model_predict(
            model=model, 
            test_dataloader=val_dataloader,
            gpu=parameters['gpu'],
            save_path=parameters['main_dir']+'model_output/val_model_output.csv',
        )
