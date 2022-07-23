import csv
import json
import random
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from mean_average_precision import MetricBuilder


def show_random_img_with_all_annotations(
    in_list: List,
    expected_list: List,
    path_to_photos: str,
    matplotlib_colours_dict: Dict,
    confidence_level: float = 0.2,
    pages: int = 5,
) -> None:
    prev = []
    for i in range(pages):
        random_img = random.randint(0, len(in_list))
        if random_img in prev:
            i -= 1
        prev.append(random_img)
        file_name = in_list[random_img]
        print(file_name)

        with cbook.get_sample_data(path_to_photos + file_name) as image_file:
            image = plt.imread(image_file)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(image, cmap="gray")

        if expected_list[random_img] != "":
            annotations = expected_list[random_img].split(" ")
            for i in range(len(annotations)):
                annotation = annotations[i].split(":")
                score = annotation[2]
                if confidence_level > float(score):
                    continue
                bbox = annotation[1].split(",")
                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                width, height = x1 - x0, y1 - y0
                cat_name = f"{annotation[0]} {round(float(score)*100,2)}%"
                rect = matplotlib.patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=1,
                    edgecolor=matplotlib_colours_dict[annotation[0]],
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x0,
                    y0,
                    cat_name,
                    fontsize=8,
                    backgroundcolor="black",
                    color=matplotlib_colours_dict[annotation[0]],
                )

        plt.show()


def calculate_map(
    dataloader: torch.utils.data.DataLoader,
    model: torchvision.models.detection.FasterRCNN,
    device: Union[torch.device, None],
) -> Dict:
    """
    Calculate AP for each clas and mean AP from torch dataloader using pytorch model
    """
    cuda_statement = torch.cuda.is_available()
    cpu_device = torch.device("cpu")
    torch.set_num_threads(1)
    # predict
    with torch.no_grad():
        f_out, f_tar = [], []
        for images, targets in dataloader:
            if cuda_statement:
                images = list(img.to(device) for img in images)
                torch.cuda.synchronize()
            else:
                images = list(img.to(cpu_device) for img in images)
            targets = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in targets
            ]
            f_tar.append(targets)
            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            f_out.append(outputs)

    # prepare model outcome
    f_out_flat = [x for xs in f_out for x in xs]
    f_tar_flat = [x for xs in f_tar for x in xs]

    pred_list, gt_list = [], []
    for i in range(len(f_out_flat)):
        # prediction
        temp_pred = []
        for ii_pred in range(len(f_out_flat[i]["boxes"].detach().numpy())):
            obj_pred = [
                int(el)
                for el in f_out_flat[i]["boxes"].detach().numpy()[ii_pred]
            ]
            obj_pred.append(
                f_out_flat[i]["labels"].detach().numpy()[ii_pred] - 1
            )
            obj_pred.append(f_out_flat[i]["scores"].detach().numpy()[ii_pred])
            temp_pred.append(obj_pred)
        pred_list.append(np.array(temp_pred))
        # ground truth
        temp_gt = []
        for ii_gt in range(len(f_tar_flat[i]["boxes"].detach().numpy())):
            obj_gt = [
                int(el)
                for el in f_tar_flat[i]["boxes"].detach().numpy()[ii_gt]
            ]
            obj_gt.append(f_tar_flat[i]["labels"].detach().numpy()[ii_gt] - 1)
            obj_gt = obj_gt + [0, 0]
            temp_gt.append(np.array(obj_gt))
        gt_list.append(np.array(temp_gt))

    # calculate metric
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=7
    )

    for i in range(len(pred_list)):
        metric_fn.add(pred_list[i], gt_list[i])

    metric = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )

    # final results
    ap_dict = {
        "photograph": metric[0.5][0]["ap"],
        "illustration": metric[0.5][1]["ap"],
        "map": metric[0.5][2]["ap"],
        "cartoon": metric[0.5][3]["ap"],
        "editorial_cartoon": metric[0.5][4]["ap"],
        "headline": metric[0.5][5]["ap"],
        "advertisement": metric[0.5][6]["ap"],
        "mAP": metric["mAP"],
    }

    return ap_dict


def dump_json(path: str, dict_to_save: Dict) -> None:
    jsonString = json.dumps(dict_to_save, indent=4)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def from_tsv_to_list(
    path: str, delimiter: str = "\n", skip_empty_lines: bool = True
) -> List:
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)
    expected = list(read_tsv)
    if skip_empty_lines:
        return [item for sublist in expected for item in sublist]
    else:
        with_empty_lines = []
        for sublist in expected:
            if len(sublist) == 0:
                with_empty_lines.append("")
            else:
                for item in sublist:
                    with_empty_lines.append(item)
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
