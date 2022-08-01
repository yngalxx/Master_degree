from typing import Dict, List, Tuple, Union

import numpy as np
from mean_average_precision import MetricBuilder

from lib.newspapersdataset import target_encoder


def prepare_eval_out_for_ap(
    output_list: List[Dict],
    target_list: List[Dict],
) -> Tuple[List[np.ndarray]]:
    """
    Prepare data into a format adequate for AP calculation
    """
    prep_pred_list, grnd_truth_list = [], []
    for i in range(len(output_list)):
        # prediction
        temp_pred = []
        for ii_pred in range(len(output_list[i]["boxes"].detach().numpy())):
            obj_pred = [
                int(el)
                for el in output_list[i]["boxes"].detach().numpy()[ii_pred]
            ]
            obj_pred.append(
                int(output_list[i]["labels"].detach().numpy()[ii_pred] - 1)
            )
            obj_pred.append(
                float(output_list[i]["scores"].detach().numpy()[ii_pred])
            )
            temp_pred.append(obj_pred)
        prep_pred_list.append(np.array(temp_pred))
        # ground truth
        temp_gt = []
        for ii_gt in range(len(target_list[i]["boxes"].detach().numpy())):
            obj_gt = [
                int(el)
                for el in target_list[i]["boxes"].detach().numpy()[ii_gt]
            ]
            obj_gt += [
                int(target_list[i]["labels"].detach().numpy()[ii_gt] - 1),
                0,
                0,
            ]
            temp_gt.append(obj_gt)
        grnd_truth_list.append(np.array(temp_gt))

    return prep_pred_list, grnd_truth_list


def prepare_data_for_ap(
    output_list: List, target_list: List, class_coding_dict: Dict
) -> Tuple[List[np.ndarray]]:
    """
    Prepare data into a format adequate for AP calculation supporting both:
    raw model output and model output stored in tsv files
    """
    p_output_list = []
    for one_image_annotations in output_list:
        temp_pred = []
        for annotation in one_image_annotations.split(" "):
            boxes = [int(box) for box in annotation.split(":")[1].split(",")]
            label = (
                int(
                    target_encoder(
                        annotation.split(":")[0],
                        class_coding_dict,
                        reverse=False,
                    )
                )
                - 1
            )
            score = float(annotation.split(":")[2])
            temp_pred.append(boxes + [label, score])

        p_output_list.append(np.array(temp_pred))

    p_ground_truth_list = []
    for one_image_annotations in target_list:
        temp_gt = []
        for annotation in one_image_annotations.split(" "):
            boxes = [int(box) for box in annotation.split(":")[1].split(",")]
            label = (
                int(
                    target_encoder(
                        annotation.split(":")[0],
                        class_coding_dict,
                        reverse=False,
                    )
                )
                - 1
            )
            difficult, crowd = 0, 0
            temp_gt.append(boxes + [label, difficult, crowd])

        p_ground_truth_list.append(np.array(temp_gt))

    return p_output_list, p_ground_truth_list


def calculate_map(
    prepared_pred_list: List[np.array],
    prepared_ground_truth_list: List[np.array],
    num_classes: int,
    class_names: List[str],
    confidence_level: Union[float, None] = None,
) -> Dict:
    """
    Calculate AP for each clas and mean AP based on preprocessed model results
    """
    if confidence_level:
        back_prepared_pred_list = prepared_pred_list.copy()
        prepared_pred_list = [
            np.array(
                [
                    elem_ii
                    for elem_ii in elem_i
                    if elem_ii[5] > confidence_level
                ]
            )
            for elem_i in back_prepared_pred_list
        ]

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=num_classes
    )

    for i in range(len(prepared_pred_list)):
        metric_fn.add(prepared_pred_list[i], prepared_ground_truth_list[i])

    metric = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )

    ap = [float(metric[0.5][i]["ap"]) for i in range(len(class_names))]
    mean_ap = sum(ap) / len(ap)

    ap_dict = {
        class_name: round(ap[i], 4)
        for (i, class_name) in enumerate(class_names)
    }
    ap_dict["mean"] = round(mean_ap, 4)

    return ap_dict
