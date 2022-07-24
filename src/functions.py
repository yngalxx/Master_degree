import csv
import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from mean_average_precision import MetricBuilder
from tqdm import tqdm


def predict_eval_set(
    dataloader: torch.utils.data.DataLoader,
    model: torchvision.models.detection.FasterRCNN,
    device: Union[torch.device, None],
) -> Tuple[List[Dict]]:
    """
    Make prediction on dataloader and return tuple with two lists containing output dictionaries
    """
    # settings
    cuda_statement = torch.cuda.is_available()
    m1_statement = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cpu_device = torch.device("cpu")
    torch.set_num_threads(1)
    # predict
    with torch.no_grad():
        f_out, f_tar = [], []
        for images, targets in tqdm(dataloader):
            if cuda_statement or m1_statement:
                images = [img.to(device) for img in images]
                if cuda_statement:
                    torch.cuda.synchronize()
            else:
                images = [img.to(cpu_device) for img in images]
            targets = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in targets
            ]
            f_tar.append(targets)
            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            f_out.append(outputs)

    # flatten list
    f_out_flat = [x for xs in f_out for x in xs]
    f_tar_flat = [x for xs in f_tar for x in xs]

    return f_out_flat, f_tar_flat


def prepare_data_for_ap(
    output_list: List[Dict],
    target_list: List[Dict],
) -> Tuple[List[np.ndarray]]:
    """
    Prepare data into a format adequate for AP calculation
    """
    prepared_pred_list, ground_truth_list = [], []
    for i in range(len(output_list)):
        # prediction
        temp_pred = []
        for ii_pred in range(len(output_list[i]["boxes"].detach().numpy())):
            obj_pred = [
                int(el)
                for el in output_list[i]["boxes"].detach().numpy()[ii_pred]
            ]
            obj_pred.append(
                output_list[i]["labels"].detach().numpy()[ii_pred] - 1
            )
            obj_pred.append(output_list[i]["scores"].detach().numpy()[ii_pred])
            temp_pred.append(obj_pred)
        prepared_pred_list.append(np.array(temp_pred))
        # ground truth
        temp_gt = []
        for ii_gt in range(len(target_list[i]["boxes"].detach().numpy())):
            obj_gt = [
                int(el)
                for el in target_list[i]["boxes"].detach().numpy()[ii_gt]
            ]
            obj_gt.append(target_list[i]["labels"].detach().numpy()[ii_gt] - 1)
            obj_gt += [0, 0]
            temp_gt.append(np.array(obj_gt))
        ground_truth_list.append(np.array(temp_gt))

    return prepared_pred_list, ground_truth_list


def calculate_map(
    prepared_pred_list: List,
    prepared_ground_truth_list: List,
    confidence_level: Union[float, None] = None,
) -> Dict:
    """
    Calculate AP for each clas and mean AP based on preprocessed model results
    """

    # remove prediction if score < confidence level we want to achieve
    if confidence_level:
        for i in range(len(prepared_pred_list)):
            for ii in range(len(prepared_pred_list[i])):
                if prepared_pred_list[i][ii][5] < confidence_level:
                    prepared_pred_list[i] = np.delete(
                        prepared_pred_list[i], ii
                    )

    # calculate metric
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=7
    )

    for i in range(len(prepared_pred_list)):
        metric_fn.add(prepared_pred_list[i], prepared_ground_truth_list[i])

    metric = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )

    return ({
        "photograph": metric[0.5][0]["ap"],
        "illustration": metric[0.5][1]["ap"],
        "map": metric[0.5][2]["ap"],
        "cartoon": metric[0.5][3]["ap"],
        "editorial_cartoon": metric[0.5][4]["ap"],
        "headline": metric[0.5][5]["ap"],
        "advertisement": metric[0.5][6]["ap"],
        "mAP": metric["mAP"]
        })


def dump_json(path: str, dict_to_save: Dict) -> None:
    jsonString = json.dumps(dict_to_save, indent=4)
    with open(path, "w") as jsonFile:
        jsonFile.write(jsonString)


def from_tsv_to_list(
    path: str, delimiter: str = "\n", skip_empty_lines: bool = True
) -> List:
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)
    expected = list(read_tsv)
    if skip_empty_lines:
        return [item for sublist in expected for item in sublist]
    with_empty_lines = []
    for sublist in expected:
        if len(sublist) == 0:
            with_empty_lines.append("")
        else:
            with_empty_lines.extend(iter(sublist))
    return with_empty_lines


def collate_fn(batch) -> Tuple:
    return tuple(zip(*batch))


def save_list_to_txt_file(path: str, output_list: List) -> None:
    with open(path, "w") as f:
        for item in output_list:
            f.write("%s\n" % item)


def save_list_to_tsv_file(path: str, output_list: List) -> None:
    with open(path, "w", newline="") as f_output:
        tsv_output = csv.writer(f_output, delimiter="\n")
        tsv_output.writerow(output_list)
