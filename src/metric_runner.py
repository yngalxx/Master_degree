import contextlib
import logging
import os
import warnings

import click
import pandas as pd

from lib.constants import General, Output
from lib.functions_catalogue import (calculate_map, from_tsv_to_list, prepare_data_for_ap)
from lib.logs import Log

# warnings
warnings.filterwarnings("ignore")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    default=General.MAIN_DIR,
    type=str,
    help="Path to the level where this repository is stored.",
    show_default=True,
)
@click.option(
    "--min_conf_level",
    default=Output.MIN_CONF_LEVEL,
    type=float,
    help=(
        "Minimum confidence level of the predictions considered in calculating"
        " the mAP metric."
    ),
    show_default=True,
)
@click.option(
    "--train_set",
    default=General.TRAIN_SET,
    type=bool,
    help="Use training data set (if files exists).",
    show_default=True,
)
@click.option(
    "--test_set",
    default=General.TEST_SET,
    type=bool,
    help="Use test data set (if files exists).",
    show_default=True,
)
@click.option(
    "--val_set",
    default=General.VAL_SET,
    type=bool,
    help="Use validation data set (if files exists).",
    show_default=True,
)
def calculate_metric(main_dir, min_conf_level, train_set, test_set, val_set):
    # check provided path
    with contextlib.redirect_stdout(logging):
        assert os.path.exists(main_dir) == True

    # initialize logger
    logger = Log("calculate_map_runer")
    logger.log_start()

    # calculate metric for train set if possible
    if train_set:
        try:
            logging.info('Processing file: "train/out.tsv"')
            out = from_tsv_to_list(f"{main_dir}/data/train/out.tsv")
            expected = from_tsv_to_list(f"{main_dir}/data/train/expected.tsv")

            pred_list, ground_truth_list = prepare_data_for_ap(
                output_list=out, target_list=expected
            )

            map_dict = calculate_map(
                pred_list, ground_truth_list, 6, min_conf_level
            )
            final_out = pd.DataFrame.from_dict(
                map_dict, orient="index", columns=["AP"]
            ).to_string()

            logging.info(f"Metric results:\n{final_out}")

        except Exception as e:
            logging.error(f"Error while processing train files: {e}")
    else:
        logging.warning(
            'Argument "train_set" not passed, calculating metric for this set'
            " is skipped"
        )

    # calculate metric for validation set if possible
    if val_set:
        try:
            logging.info('Processing file: "dev-0/out.tsv"')
            out = from_tsv_to_list(f"{main_dir}/data/dev-0/out.tsv")
            expected = from_tsv_to_list(f"{main_dir}/data/dev-0/expected.tsv")

            pred_list, ground_truth_list = prepare_data_for_ap(
                output_list=out, target_list=expected
            )

            map_dict = calculate_map(
                pred_list, ground_truth_list, 6, min_conf_level
            )
            final_out = pd.DataFrame.from_dict(
                map_dict, orient="index", columns=["AP"]
            ).to_string()

            logging.info(f"Metric results:\n{final_out}")

        except Exception as e:
            logging.error(f"Error while processing validation files: {e}")
    else:
        logging.warning(
            'Argument "val_set" not passed, calculating metric for this set is'
            " skipped"
        )

    # calculate metric for test set if possible
    if test_set:
        try:
            logging.info('Processing file: "test-A/out.tsv"')
            out = from_tsv_to_list(f"{main_dir}/data/test-A/out.tsv")
            expected = from_tsv_to_list(f"{main_dir}/data/test-A/expected.tsv")

            pred_list, ground_truth_list = prepare_data_for_ap(
                output_list=out, target_list=expected
            )

            map_dict = calculate_map(
                pred_list, ground_truth_list, 6, min_conf_level
            )
            final_out = pd.DataFrame.from_dict(
                map_dict, orient="index", columns=["AP"]
            ).to_string()

            logging.info(f"Metric results:\n{final_out}")

        except Exception as e:
            logging.error(f"Error while processing test files: {e}")
    else:
        logging.warning(
            'Argument "test_set" not passed, calculating metric for this set'
            " is skipped"
        )

    if not train_set and not val_set and not test_set:
        logging.error(
            'None of the arguments: "train_set", "val_set" and "test_set"'
            " passed, aborted!"
        )
        raise ValueError()

    # end logger
    logger.log_end()


if __name__ == "__main__":
    calculate_metric()
