import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from newspapersdataset import NewspapersDataset
from newspapersdataset import prepare_data_for_dataloader
from train_model import train_model
from test_model import model_predict
from functions import from_tsv_to_list
from functions import collate_fn
import warnings

# warnings
warnings.filterwarnings("ignore")


def controller(
    channel: int, 
    num_classes: int, 
    learning_rate: float, 
    batch_size: int, 
    num_epochs: int, 
    rescale: str, 
    shuffle: bool, 
    weight_decay: float, 
    lr_scheduler: bool, 
    lr_step_size: int, 
    lr_gamma: float, 
    trainable_backbone_layers: int, 
    num_workers: int, 
    main_dir: str, 
    image_dir: str, 
    annotations_dir: str, 
    train: bool, 
    predict: bool, 
    train_set: bool, 
    test_set: bool, 
    val_set: bool, 
    gpu: bool, 
    bbox_format: str
) -> None: 
    try:
        rescale = float(rescale)
    except:
        rescale = rescale.split('/')
        rescale = [int(rescale[0]), int(rescale[1])]

    # read data and create dataloaders 
    data_transform = T.Compose([
        T.Grayscale(num_output_channels=channel),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
        ])

    if val_set:
        # create validation data loader
        expected_val = from_tsv_to_list(annotations_dir+'dev-0/expected.tsv')
        in_val = from_tsv_to_list(annotations_dir+'dev-0/in.tsv')
        val_paths = [image_dir+path for path in in_val]
        data_val = prepare_data_for_dataloader(
            img_dir=image_dir,
            in_list=in_val,
            expected_list=expected_val,
            bbox_format=bbox_format,
            scale=rescale,
            test=False,
            )
        dataset_val = NewspapersDataset(
            df=data_val,
            images_path=val_paths,
            scale=rescale,
            transforms=data_transform,
            test=False,
            )
        val_dataloader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            )
    else:
        val_dataloader=None

    if train_set:
        # create train data loader
        expected_train = from_tsv_to_list(annotations_dir+'train/expected.tsv')
        in_train = from_tsv_to_list(annotations_dir+'train/in.tsv')
        train_paths = [image_dir+path for path in in_train]
        data_train = prepare_data_for_dataloader(
            img_dir=image_dir,
            in_list=in_train,
            expected_list=expected_train,
            bbox_format=bbox_format,
            scale=rescale,
            test=False,
            )
        dataset_train = NewspapersDataset(
            df=data_train,
            images_path=train_paths,
            scale=rescale,
            transforms=data_transform,
            test=False,
            )
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        # train phase
        if train:
            # pre-trained resnet50 model
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                trainable_backbone_layers=trainable_backbone_layers
            )

            # replace the pre-trained head with a new one
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, 
                num_classes=num_classes
                )

            # optimizer
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            # learning rate scheduler decreases the learning rate by 'gamma' every 'step_size'
            if lr_scheduler:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=lr_step_size,
                    gamma=lr_gamma
                )
            else:
                lr_scheduler = None

            # train and save model
            trained_model = train_model(
                pre_treined_model=model, 
                optimizer=optimizer, 
                train_dataloader=train_dataloader,
                epochs=num_epochs, 
                gpu=gpu,
                val_dataloader=val_dataloader, 
                lr_scheduler=lr_scheduler, 
            )
            torch.save(trained_model, main_dir+'saved_models/model.pth')

    # prediction phase
    if predict:
        if torch.cuda.is_available():
            model = torch.load(main_dir+'saved_models/model.pth')
        else:
            model = torch.load(main_dir+'saved_models/model.pth', map_location=torch.device('cpu'))
        if test_set:
            # create test data loader
            in_test = from_tsv_to_list(annotations_dir+'test-A/in.tsv')
            test_paths = [image_dir+path for path in in_test]
            data_test = prepare_data_for_dataloader(
                img_dir=image_dir,
                in_list=in_test,
                expected_list=None,
                bbox_format=bbox_format,
                scale=rescale,
                test=True,
                )
            dataset_test = NewspapersDataset(
                df=data_test,
                images_path=test_paths,
                scale=rescale,
                transforms=data_transform,
                test=True,
                )
            test_dataloader = DataLoader(
                dataset_test,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
            # prediction on test set
            print('###  Evaluating test set  ###')
            model_predict(
                model=model, 
                dataloader=test_dataloader,
                gpu=gpu,
                save_path=main_dir+'model_output/test_model_output.csv',
            )

        # prediction on train set (to check under/overfitting)
        if train_set:
            print('###  Evaluating train set  ###')
            model_predict(
                model=model, 
                dataloader=train_dataloader,
                gpu=gpu,
                save_path=main_dir+'model_output/train_model_output.csv',
            )
            
        # prediction on validation set
        if val_set:
            print('###  Evaluating validation set  ###')
            model_predict(
                model=model, 
                dataloader=val_dataloader,
                gpu=gpu,
                save_path=main_dir+'model_output/val_model_output.csv',
            )

