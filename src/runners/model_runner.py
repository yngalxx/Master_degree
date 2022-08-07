import logging
import os

import click
from constants import Data, General, Model, Output

from lib.logs import Log
from lib.model import (check_model_and_save, create_model_config,
                       evaluate_model, initalize_model, initialize_optimizer,
                       load_model_state_dict, train_model)
from lib.newspapers_dataset import create_dataloader
from lib.save_load_data import from_tsv_to_list


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--channel",
    default=Data.CHANNEL,
    type=int,
    help="Image channels: 3 <= RGB, 1 <= greyscale.",
    show_default=True,
)
@click.option(
    "--num_classes",
    default=Data.NUM_CLASSES,
    type=int,
    help="Number of classes.",
    show_default=True,
)
@click.option(
    "--learning_rate",
    default=Model.LEARNING_RATE,
    type=float,
    help="Learning rate value.",
    show_default=True,
)
@click.option(
    "--batch_size",
    default=Model.BATCH_SIZE,
    type=int,
    help="Number of batches.",
    show_default=True,
)
@click.option(
    "--num_epochs",
    default=Model.NUM_EPOCHS,
    type=int,
    help="Number of epochs.",
    show_default=True,
)
@click.option(
    "--rescale",
    default=Data.RESCALE,
    type=str,
    help=(
        "2 possible ways to rescale your images and also annotations. First"
        ' one is by using following pattern "width/height" and then each image'
        " will be scaled to that size, thanks to it you will have every image"
        " in the same size (less computational complexity). The other way is"
        " to enter a float value (however you still have to put it in a string"
        ' i.e. ".5" and value has to be bigger than 0 and smaller or equal'
        " than 1), then each image will be multiplied by this value. If you"
        " pass 1 as a value images and annotations will not be scaled."
    ),
    show_default=True,
)
@click.option(
    "--shuffle",
    default=Data.SHUFFLE,
    type=bool,
    help="Shuffle data.",
    show_default=True,
)
@click.option(
    "--weight_decay",
    default=Model.WEIGHT_DECAY,
    type=float,
    help="Weight decay regularization value.",
    show_default=True,
)
@click.option(
    "--lr_scheduler",
    default=Model.LR_SCHEDULER,
    type=bool,
    help=(
        "Learning rate scheduler: if value=True learning rate scheduler will"
        " be enabled, if value=False it will be disabled."
    ),
    show_default=True,
)
@click.option(
    "--lr_step_size",
    default=Model.LR_STEP_SIZE,
    type=int,
    help=(
        "Step size of learning rate scheduler: valid only if learning rate"
        " scheduler is enabled."
    ),
    show_default=True,
)
@click.option(
    "--lr_gamma",
    default=Model.LR_GAMMA,
    type=float,
    help=(
        "Valid only when learning rate scheduling is enabled, passed value "
        "determines the learning rate multiplier."
    ),
    show_default=True,
)
@click.option(
    "--trainable_backbone_layers",
    default=Model.TRAINABLE_BACKBONE_LAYER,
    type=int,
    help="Number of trainable layers in pretrained ResNet-50 network.",
    show_default=True,
)
@click.option(
    "--pretrained",
    default=Model.PRETRAINED,
    type=bool,
    help=(
        "Start training using pretrained ResNet-50 instead of training from"
        " scratch"
    ),
    show_default=True,
)
@click.option(
    "--num_workers",
    default=Data.NUM_WORKERS,
    type=int,
    help=(
        "Setting the argument num_workers as a positive integer will turn on"
        " multi-process data loading with the specified number of loader"
        " worker processes."
    ),
    show_default=True,
)
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Working directory path.",
    show_default=True,
)
@click.option(
    "--train_set",
    default=General.TRAIN_SET,
    type=bool,
    help="Use training data set.",
    show_default=True,
)
@click.option(
    "--test_set",
    default=General.TEST_SET,
    type=bool,
    help="Use test data set.",
    show_default=True,
)
@click.option(
    "--val_set",
    default=General.VAL_SET,
    type=bool,
    help=(
        "Use validation data set. If value is equal to False, the evaluation"
        " during training will not be available, nor model will be saved"
        " afterwards. if you want to save it anyway use the"
        " '--force_save_model' flag."
    ),
    show_default=True,
)
@click.option(
    "--gpu",
    default=General.GPU,
    type=bool,
    help="Enable training on GPU.",
    show_default=True,
)
@click.option(
    "--bbox_format",
    default=Data.BBOX_FORMAT,
    type=str,
    help=(
        'Bounding boxes format. Other allowed format is "x0y0wh", where w -'
        " width and h - height."
    ),
    show_default=True,
)
@click.option(
    "--force_save_model",
    default=Output.FORCE_SAVE_MODEL,
    type=bool,
    help="Force save model despite of its performance.",
    show_default=True,
)
@click.option(
    "--val_map_threshold",
    default=Output.VAL_MAP_THRESHOLD,
    type=float,
    help=(
        "Value of the mAP evaluation metric, after which training will be"
        " stopped. If the model exceeds the set threshold, it will be saved"
        " without considering the previous model results as a comparison (in"
        " this situation, the force_save_model argument will automatically be"
        " set to True)."
    ),
    show_default=True,
)
@click.option(
    "--train",
    type=bool,
    help="Model training enabled",
    required=True,
)
@click.option(
    "--evaluate",
    type=bool,
    help="Model evaluation enabled",
    required=True,
)
def model_runner(
    channel,
    num_classes,
    learning_rate,
    batch_size,
    num_epochs,
    rescale,
    shuffle,
    weight_decay,
    lr_scheduler,
    lr_step_size,
    lr_gamma,
    trainable_backbone_layers,
    num_workers,
    main_dir,
    train,
    evaluate,
    train_set,
    test_set,
    val_set,
    gpu,
    bbox_format,
    force_save_model,
    pretrained,
    val_map_threshold,
):
    # initialize logger
    logger = Log("model_runner", main_dir)
    logger.log_start()

    if not train:
        logging.warning(
            'Argument "train" not passed, this phase will be skipped'
        )

    if not evaluate:
        logging.warning(
            'Argument "evaluate" not passed, this phase will be skipped'
        )

    if not train and not evaluate:
        logging.error(
            'Both arguments "train" and "evaluate" not passed, code will be'
            " forced to quit!"
        )
        raise ValueError()

    if not train_set and not val_set and not test_set:
        logging.error(
            'None of the arguments: "train_set", "val_set" and "test_set"'
            " passed, code will be forced to quit!"
        )
        raise ValueError()

    # check provided path
    scraped_photos_dir = f"{main_dir}/scraped_photos/"
    assert os.path.exists(scraped_photos_dir) == True
    annotations_dir = f"{main_dir}/data/"
    assert os.path.exists(annotations_dir) == True

    try:
        if "/" in rescale:
            rescale = rescale.split("/")
            rescale = [int(rescale[0]), int(rescale[1])]
        else:
            rescale = float(rescale)
    except:
        logging.error(f"Wrong 'rescale' argument value '{rescale}'")
        raise ValueError()

    if val_set:
        # create validation dataloader
        logging.info("Creating validation dataloader")
        try:
            path = "dev-0/expected.tsv"
            expected_val = from_tsv_to_list(f"{annotations_dir}{path}")

            path = "dev-0/in.tsv"
            in_val = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()

        val_dataloader = create_dataloader(
            image_dir=scraped_photos_dir,
            in_list=in_val,
            expected_list=expected_val,
            class_coding_dict=Data.CLASS_CODING_DICT,
            bbox_format=bbox_format,
            rescale=rescale,
            test=False,
            channel=channel,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        val_dataloader = None

    config_dir_name = "model_config/"

    if train_set:
        # create train data loader
        logging.info("Creating train dataloader")
        try:
            path = "train/expected.tsv"
            expected_train = from_tsv_to_list(f"{annotations_dir}{path}")

            path = "train/in.tsv"
            in_train = from_tsv_to_list(f"{annotations_dir}{path}")
        except:
            logging.exception(
                f"File '{path}' not found, code will be forced to quit"
            )
            raise FileNotFoundError()

        train_dataloader = create_dataloader(
            image_dir=scraped_photos_dir,
            in_list=in_train,
            expected_list=expected_train,
            class_coding_dict=Data.CLASS_CODING_DICT,
            bbox_format=bbox_format,
            rescale=rescale,
            test=False,
            channel=channel,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # train phase
        if train:
            logging.info("Model training phase - started!")
            model_path = f"{main_dir}/{config_dir_name}"
            if not os.path.exists(model_path):
                logging.info(
                    f"Directory '{config_dir_name}' doesn't exist, creating one"
                )
                os.makedirs(model_path)

            # store model related arguments in config file for future predictions etc.
            model_config = create_model_config(
                channel=channel,
                num_classes=num_classes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                rescale=rescale,
                shuffle=shuffle,
                weight_decay=weight_decay,
                lr_scheduler=lr_scheduler,
                lr_step_size=lr_step_size,
                lr_gamma=lr_gamma,
                trainable_backbone_layers=trainable_backbone_layers,
                num_workers=num_workers,
                gpu=gpu,
                bbox_format=bbox_format,
                pretrained=pretrained,
            )

            # model initialization
            model = initalize_model(
                pretrained=pretrained,
                trainable_backbone_layers=trainable_backbone_layers,
                num_classes=num_classes,
            )

            # optimizer and learning rate scheduler (if enabled)
            optimizer, lrs = initialize_optimizer(
                torch_model_parameters=model.parameters(),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                lr_scheduler=lr_scheduler,
                lr_step_size=lr_step_size,
                lr_gamma=lr_gamma,
            )

            # train and save model
            trained_model, eval_df, force_save_model = train_model(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                epochs=num_epochs,
                gpu=gpu,
                val_dataloader=val_dataloader,
                lr_scheduler=lrs,
                num_classes=num_classes - 1,
                class_names=Data.CLASS_NAMES,
                val_map_threshold=val_map_threshold,
                force_save_model=force_save_model,
            )

            # check if model can be saved and save it
            evaluate = check_model_and_save(
                model_path=model_path,
                evaluation_df=eval_df,
                val_dataloader=val_dataloader,
                force_save_model=force_save_model,
                model_config=model_config,
                trained_model_state_dict=trained_model.state_dict(),
            )

    # evaluation phase
    if evaluate:
        logging.info("Model evaluation phase - started!")
        # initialize model
        model = initalize_model(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            num_classes=num_classes,
        )

        # load model state dict
        try:
            model = load_model_state_dict(
                gpu=gpu,
                init_model=model,
                config_dir_path=f"{main_dir}/{config_dir_name}/",
            )
        except:
            logging.exception(
                f"No model found in '{config_dir_name}' directory, code will"
                " be forced to quit"
            )
            raise FileNotFoundError()

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

            test_dataloader = create_dataloader(
                image_dir=scraped_photos_dir,
                in_list=in_test,
                expected_list=None,
                class_coding_dict=Data.CLASS_CODING_DICT,
                bbox_format=bbox_format,
                rescale=rescale,
                test=True,
                channel=channel,
                batch_size=batch_size,
                shuffle=shuffle,
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
                expected_list_exists=Output.TEST_SET_EXPECTED,
                num_classes=num_classes - 1,
                class_names=Data.CLASS_NAMES,
                class_coding_dict=Data.CLASS_CODING_DICT,
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
                expected_list_exists=True,
                num_classes=num_classes - 1,
                class_names=Data.CLASS_NAMES,
                class_coding_dict=Data.CLASS_CODING_DICT,
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
                expected_list_exists=True,
                num_classes=num_classes - 1,
                class_names=Data.CLASS_NAMES,
                class_coding_dict=Data.CLASS_CODING_DICT,
            )

    # end logger
    logger.log_end()


if __name__ == "__main__":
    model_runner()
