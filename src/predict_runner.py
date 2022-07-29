import json
import os

import click
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from constants import Output_args
from functions_catalogue import (collate_fn, predict_one_img,
                                 show_random_img_with_all_annotations)
from newspapersdataset import NewspapersDataset, prepare_data_for_dataloader


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--path_to_image",
    type=str,
    help="Path to the image on which you want to make prediciton.",
    required=True,
)
@click.option(
    "--model_config_path",
    type=str,
    default=Output_args.MODEL_CONFIG_PATH,
    help="Path to directory containing model and json config file.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    type=float,
    default=Output_args.MIN_CONF_LEVEL,
    help="Minimum confidence level for model predictions to show up.",
    show_default=True,
)
def predict(path_to_image, model_config_path, min_conf_level):
    # check provided path
    assert os.path.exists(path_to_image) == True

    # extract path and file name
    image_path_split = path_to_image.split("/")
    image_dir = f'{"/".join(image_path_split[:-1])}/'
    image_name = image_path_split[-1]

    # read model config file
    try:
        path = "model_config.json"
        config = json.load(open(f"{model_config_path}/{path}"))
    except:
        raise FileNotFoundError(
            f"File '{path}' not found, code will be forced to quit"
        )

    if isinstance(config["rescale"], float):
        rescale = config["rescale"]
    else:
        rescale = [config["rescale"][0], config["rescale"][1]]

    data_transform = T.Compose(
        [
            T.Grayscale(num_output_channels=config["channel"]),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    # create torch dataloader
    data = prepare_data_for_dataloader(
        img_dir=image_dir,
        in_list=[image_name],
        bbox_format=config["bbox_format"],
        scale=rescale,
        test=True,
    )
    dataset = NewspapersDataset(
        df=data,
        images_path=[path_to_image],
        scale=rescale,
        transforms=data_transform,
        test=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=config["pretrained"],
        trainable_backbone_layers=config["trainable_backbone_layers"],
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=config["num_classes"]
    )
    try:
        model.load_state_dict(
            torch.load(
                f"{model_config_path}/model.pth",
                map_location=torch.device("cpu"),
            ),
            strict=True,
        )
    except:
        raise FileNotFoundError(
            f"No model found in '{config_dir_name}' directory, code will be"
            " forced to quit"
        )

    model.eval()

    pred = predict_one_img(
        model=model,
        dataloader=dataloader,
        image_name=image_name,
        path_to_image=image_dir,
    )

    show_random_img_with_all_annotations(
        in_list=[image_name],
        expected_list=pred,
        path_to_photos=image_dir,
        confidence_level=min_conf_level,
        matplotlib_colours_dict={
            "photograph": "lime",
            "illustration": "orangered",
            "map": "yellow",
            "cartoon": "deepskyblue",
            "headline": "cyan",
            "advertisement": "deeppink",
        },
        jupyter=False,
        pages=1,
    )


if __name__ == "__main__":
    predict()