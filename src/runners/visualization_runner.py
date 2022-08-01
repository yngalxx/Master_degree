import os

import click

from lib.save_load_data import from_tsv_to_list
from lib.visualization import show_random_img_with_all_annotations

from constants import Data, General, Output


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Path to the level where this repository is stored.",
    show_default=True,
)
@click.option(
    "--examples",
    default=Output.EXAMPLES,
    type=int,
    help="Number of random examples to show up.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    default=Output.MIN_CONF_LEVEL,
    type=float,
    help="Minimum confidence level for model predictions to show up.",
    show_default=True,
)
def visualizer(main_dir, examples, min_conf_level):
    # check provided path
    path_to_photos = f"{main_dir}/scraped_photos/"
    assert os.path.exists(path_to_photos) == True
    data_path = f"{main_dir}/data/test-A/"
    assert os.path.exists(data_path) == True

    # read data
    try:
        in_test_file = "in.tsv"
        in_test_full_path = f"{data_path}{in_test_file}"
        in_test = from_tsv_to_list(in_test_full_path)
    except:
        raise FileNotFoundError(
            f"File '{in_test_file}' not found, code will be forced to quit"
        )
    try:
        out_test_file = "out.tsv"
        test_out_full_path = f"{data_path}{out_test_file}"
        out_list = from_tsv_to_list(test_out_full_path, skip_empty_lines=False)
    except:
        raise FileNotFoundError(
            f"File '{out_test_file}' not found, code will be forced to quit"
        )

    show_random_img_with_all_annotations(
        in_list=in_test,
        expected_list=out_list,
        path_to_photos=path_to_photos,
        matplotlib_colours_dict=Data.CLASS_COLORS_DICT,
        confidence_level=min_conf_level,
        pages=examples,
    )


if __name__ == "__main__":
    visualizer()
