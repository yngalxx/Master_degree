import json
import os

import click

from constants import Data, Output

from lib.newspapers_dataset import create_dataloader
from lib.model import initalize_model, load_model_state_dict, predict_one_img
from lib.visualization import show_random_img_with_all_annotations


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--path_to_image",
    type=str,
    help=(
        "Path to the image on which you want to make prediciton. The image"
        " name must match the following pattern: 'number.(jpg, png or other)',"
        " e.g. '1.jpg'."
    ),
    required=True,
)
@click.option(
    "--model_config_path",
    default=Output.MODEL_CONFIG_PATH,
    type=str,
    help="Path to directory containing model and json config file.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    default=Output.MIN_CONF_LEVEL,
    type=float,
    help="Minimum confidence level for model predictions to show up.",
    show_default=True,
)
def predict(path_to_image, model_config_path, min_conf_level):
    # check provided path
    assert os.path.exists(path_to_image) == True
    assert os.path.exists(model_config_path) == True

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

    # create torch dataloader
    dataloader=create_dataloader(image_dir=image_dir, in_list=[image_name], expected_list=None, class_coding_dict=Data.CLASS_CODING_DICT, bbox_format=config["bbox_format"], rescale=rescale, test=True, channel=config["channel"], batch_size=1, shuffle=False, num_workers=2)
    
    model = initalize_model(pretrained=config["pretrained"], trainable_backbone_layers=config["trainable_backbone_layers"], num_classes=config["num_classes"])

    try:
        model = load_model_state_dict(gpu=config["gpu"], init_model=model, config_dir_path=f'{model_config_path}/')
    except:
            raise FileNotFoundError(f"No model found in '{model_config_path.split('/')[-1]}' directory, code will be forced to quit")

    pred = predict_one_img(
        model=model,
        dataloader=dataloader,
        image_name=image_name,
        path_to_image=image_dir,
        class_coding_dict=Data.CLASS_CODING_DICT,
    )

    show_random_img_with_all_annotations(
        in_list=[image_name],
        expected_list=pred,
        path_to_photos=image_dir,
        confidence_level=min_conf_level,
        matplotlib_colours_dict=Data.CLASS_COLORS_DICT,
        pages=1,
    )


if __name__ == "__main__":
    predict()
