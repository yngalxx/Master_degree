import os
import warnings
import logging
import torch
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from functions_catalogue import collate_fn, from_tsv_to_list, dump_json
from evaluate_model import evaluate_model
from newspapersdataset import NewspapersDataset, prepare_data_for_dataloader
from train_model import train_model


# warnings
warnings.filterwarnings("ignore")


def model_pipeline(
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
    train: bool,
    evalutaion: bool,
    train_set: bool,
    test_set: bool,
    val_set: bool,
    gpu: bool,
    bbox_format: str,
) -> None:
    scraped_photos_dir = f"{main_dir}scraped_photos/"
    annotations_dir = f"{main_dir}data/"

    try:
        rescale = float(rescale)
    except BaseException:
        rescale = rescale.split("/")
        rescale = [int(rescale[0]), int(rescale[1])]

    # read data and create dataloaders
    data_transform = T.Compose(
        [
            T.Grayscale(num_output_channels=channel),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    if val_set:
        # create validation dataloader
        logging.info('Creating validation dataloader')
        expected_val = from_tsv_to_list(f"{annotations_dir}dev-0/expected.tsv")
        in_val = from_tsv_to_list(f"{annotations_dir}dev-0/in.tsv")
        val_paths = [scraped_photos_dir + path for path in in_val]
        data_val = prepare_data_for_dataloader(
            img_dir=scraped_photos_dir,
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
        val_dataloader = None

    if train_set:
        # create train data loader
        logging.info('Creating train dataloader')
        expected_train = from_tsv_to_list(
            f"{annotations_dir}train/expected.tsv"
        )
        in_train = from_tsv_to_list(f"{annotations_dir}train/in.tsv")
        train_paths = [scraped_photos_dir + path for path in in_train]
        data_train = prepare_data_for_dataloader(
            img_dir=scraped_photos_dir,
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
            model_path = f"{main_dir}model_config/"
            if not os.path.exists(model_path):
                logging.info('Directory "model_config" doesn\'t exist, creating one')
                os.makedirs(model_path)
            
            # save rescale, bbox_format and channel arguments in config file for future predictions
            model_config = {
                'rescale': rescale,
                'bbox_format': bbox_format,
                'channel': channel,
                'num_classes': num_classes,
                'trainable_backbone_layers': trainable_backbone_layers
            }
            dump_json(f'{model_path}model_config.json', model_config)

            # pre-trained resnet50 model
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                trainable_backbone_layers=trainable_backbone_layers,
            )

            # replace the pre-trained head with a new one
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes=num_classes
            )

            # optimizer
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            # learning rate scheduler decreases the learning rate by 'gamma' every 'step_size'
            if lr_scheduler:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=lr_step_size, gamma=lr_gamma
                )
            else:
                lr_scheduler = None

            # train and save model
            logging.info('Model training - started!')
            trained_model = train_model(
                pre_treined_model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                epochs=num_epochs,
                gpu=gpu,
                val_dataloader=val_dataloader,
                lr_scheduler=lr_scheduler,
            )
            torch.save(trained_model.state_dict(), f"{model_path}model.pth")
            logging.info('Model weights saved in "model_config" directory')

    # evalutaion phase
    if evalutaion:
        # initialize model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=trainable_backbone_layers)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=num_classes
        )
        # load model state dict
        if torch.cuda.is_available() and gpu:
            map_location=torch.device(torch.cuda.current_device())
        else:
            map_location=torch.device("cpu")
        
        try:
            model.load_state_dict(torch.load(f"{main_dir}model_config/model.pth", map_location=map_location), strict=True)
        except:
            logging.exception("No model found, code will be forced to quit")
            raise Exception()

        model.eval()
        logging.info(f'Model loaded correctly')

        if test_set:
            # create test data loader
            logging.info('Creating test dataloader')
            in_test = from_tsv_to_list(f"{annotations_dir}test-A/in.tsv")
            test_paths = [scraped_photos_dir + path for path in in_test]
            data_test = prepare_data_for_dataloader(
                img_dir=scraped_photos_dir,
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
            # evaluation on test set
            logging.info('Test set evaluation - started!')
            evaluate_model(
                model=model,
                dataloader=test_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="test-A",
                test=False,
            )

        # evaluation on train set (to check under/overfitting)
        if train_set:
            logging.info('Train set evaluation - started!')
            evaluate_model(
                model=model,
                dataloader=train_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="train",
                test=True,
            )

        # evaluation on validation set
        if val_set:
            logging.info('Validation set evaluation - started!')
            evaluate_model(
                model=model,
                dataloader=val_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="dev-0",
                test=True,
            )
