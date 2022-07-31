import logging
import os
import warnings

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lib.evaluate_model import evaluate_model
from lib.functions_catalogue import collate_fn, dump_json, from_tsv_to_list
from lib.newspapersdataset import (NewspapersDataset,
                                   prepare_data_for_dataloader)
from lib.train_model import train_model

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
    evaluate: bool,
    train_set: bool,
    test_set: bool,
    val_set: bool,
    gpu: bool,
    bbox_format: str,
    force_save_model: str,
    pretrained: bool,
    val_map_threshold: float,
) -> None:
    # check provided path
    scraped_photos_dir = f"{main_dir}/scraped_photos/"
    assert os.path.exists(scraped_photos_dir) == True
    annotations_dir = f"{main_dir}/data/"
    assert os.path.exists(annotations_dir) == True

    config_dir_name = "model_config/"

    try:
        if "/" in rescale:
            rescale = rescale.split("/")
            rescale = [int(rescale[0]), int(rescale[1])]
        else:
            rescale = float(rescale)
    except:
        logging.error(f"Wrong 'rescale' argument value '{rescale}'")
        raise ValueError()

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
        logging.info("Creating validation dataloader")
        try:
            path = "dev-0/expected.tsv"
            expected_val = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()
        try:
            path = "dev-0/in.tsv"
            in_val = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()

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
        logging.info("Creating train dataloader")
        try:
            path = "train/expected.tsv"
            expected_train = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()
        try:
            path = "train/in.tsv"
            in_train = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()

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
            logging.info("Model training phase - started!")
            model_path = f"{main_dir}/{config_dir_name}"
            if not os.path.exists(model_path):
                logging.info(
                    f"Directory {config_dir_name} doesn't exist, creating one"
                )
                os.makedirs(model_path)

            # save model related arguments in config file for future predictions etc.
            model_config = {
                "channel": channel,
                "num_classes": num_classes,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "rescale": rescale,
                "shuffle": shuffle,
                "weight_decay": weight_decay,
                "lr_scheduler": lr_scheduler,
                "lr_step_size": lr_step_size,
                "lr_gamma": lr_gamma,
                "trainable_backbone_layers": trainable_backbone_layers,
                "num_workers": num_workers,
                "gpu": gpu,
                "bbox_format": bbox_format,
                "pretrained": pretrained,
            }

            # pre-trained resnet50 model
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
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
            trained_model, eval_df, force_save_model = train_model(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                epochs=num_epochs,
                gpu=gpu,
                val_dataloader=val_dataloader,
                lr_scheduler=lr_scheduler,
                num_classes=num_classes - 1,
                val_map_threshold=val_map_threshold,
                force_save_model=force_save_model,
            )

            # check if model can be saved
            model_metric_path = f"{model_path}model_eval_metric.csv"
            if val_dataloader and not force_save_model:
                if not os.path.exists(model_metric_path):
                    logging.info(
                        "Previous model validation results not found, model"
                        " will be saved"
                    )
                    force_save_model = True
                else:
                    prev_eval_df = pd.read_csv(model_metric_path, index_col=0)
                    prev_results_map, current_results_map = (
                        prev_eval_df["AP"]["mean"],
                        eval_df["AP"]["mean"],
                    )
                    if current_results_map >= prev_results_map:
                        logging.info(
                            "Model validation results (mAP ="
                            f" {current_results_map:.4f}) are better than"
                            f" previous ones (mAP = {prev_results_map:.4f}),"
                            " previous model will be overridden"
                        )
                        force_save_model = True
                    else:
                        logging.info(
                            "Model validation results (mAP ="
                            f" {current_results_map:.4f}) are worse than"
                            f" previous ones (mAP = {prev_results_map:.4f}),"
                            " model will not be saved and evaluation phase"
                            " will be skipped"
                        )
                        evaluate = False

            # save model state dict, config json and metric dataframe
            if force_save_model:
                dump_json(f"{model_path}model_config.json", model_config)
                logging.info(
                    f'Model config json saved in "{config_dir_name}" directory'
                )
                torch.save(
                    trained_model.state_dict(), f"{model_path}model.pth"
                )
                logging.info(f'Model saved in "{config_dir_name}" directory')

                if val_dataloader:
                    eval_df.to_csv(model_metric_path)
                    logging.info(
                        "Model validation results saved in"
                        f' "{config_dir_name}" directory'
                    )

                if not val_dataloader and os.path.exists(model_metric_path):
                    os.remove(model_metric_path)

    # evaluation phase
    if evaluate:
        logging.info("Model evaluation phase - started!")
        # initialize model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=num_classes
        )
        # load model state dict
        if torch.cuda.is_available() and gpu:
            map_location = torch.device(torch.cuda.current_device())
        else:
            map_location = torch.device("cpu")

        try:
            model.load_state_dict(
                torch.load(
                    f"{main_dir}/{config_dir_name}model.pth",
                    map_location=map_location,
                ),
                strict=True,
            )
        except:
            logging.exception(
                f"No model found in '{config_dir_name}' directory, code will"
                " be forced to quit"
            )
            raise FileNotFoundError()

        model.eval()
        logging.info("Model loaded correctly")

        if test_set:
            # create test data loader
            logging.info("Creating test dataloader")
            try:
                path = "test-A/in.tsv"
                in_test = from_tsv_to_list(f"{annotations_dir}{path}")
            except:
                logging.exception(
                    f"File '{path}' not found, code will be forced to quit"
                )
                raise FileNotFoundError()

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
            logging.info("Test set evaluation - started!")
            evaluate_model(
                model=model,
                dataloader=test_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="test-A",
                test=True,
                num_classes=num_classes - 1,
            )

        # evaluation on train set (to check under/overfitting)
        if train_set:
            logging.info("Train set evaluation - started!")
            evaluate_model(
                model=model,
                dataloader=train_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="train",
                test=True,
                num_classes=num_classes - 1,
            )

        # evaluation on validation set
        if val_set:
            logging.info("Validation set evaluation - started!")
            evaluate_model(
                model=model,
                dataloader=val_dataloader,
                gpu=gpu,
                main_dir=main_dir,
                save_path="dev-0",
                test=True,
                num_classes=num_classes - 1,
            )
