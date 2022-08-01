from typing import Dict, List

import imagesize
import numpy as np

from lib.newspapersdataset import target_encoder


def parse_model_outcome(
    img_names_list: List[str],
    image_directory: str,
    new_sizes_list: List,
    predicted_labels_list: List[int],
    scores_list: List[float],
    predicted_bboxes_list: List[int],
    class_coding_dict: Dict,
) -> List[str]:
    """
    Parse model outcome to list (list of lists where 1 element contains all
    annotations for 1 image)
    """
    out_list = []
    for i in range(len(img_names_list)):
        img_width, img_height = imagesize.get(
            f"{image_directory}{img_names_list[i]}"
        )
        out_str = ""
        for ii in range(len(predicted_labels_list[i])):
            scaler_width = np.float(img_width) / np.float(
                new_sizes_list[i][0][0]
            )
            scaler_height = np.float(img_height) / np.float(
                new_sizes_list[i][0][1]
            )
            label = target_encoder(
                int(predicted_labels_list[i][ii]),
                class_coding_dict,
                reverse=True,
            )
            x0 = str(
                int(round(predicted_bboxes_list[i][ii][0], 0) * scaler_width)
            )
            y0 = str(
                int(round(predicted_bboxes_list[i][ii][1], 0) * scaler_height)
            )
            x1 = str(
                int(round(predicted_bboxes_list[i][ii][2], 0) * scaler_width)
            )
            y1 = str(
                int(round(predicted_bboxes_list[i][ii][3], 0) * scaler_height)
            )

            out_str = (
                f"{out_str}{label}:{x0},{y0},{x1},{y1}:{scores_list[i][ii]} "
            )

        out_str = out_str.strip(" ")
        out_list.append(out_str)

    return out_list
