import sys
import csv
import pathlib
import json
from typing import Dict, List, Tuple, Union
import pandas as pd
import hashlib
import numpy as np
import torch
import torchvision
from mean_average_precision import MetricBuilder
from tqdm import tqdm
import random
from matplotlib.patches import Rectangle
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt

try:
    sys.path.append(
        "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])
    )
    from image_size import get_image_size # source: https://github.com/scardine/image_size
except:
    raise ImportWarning('Repository "image_size" not found, code will crash if it is needed during execution')


def target_encoder(label: int, reverse=False):
    label_dict = {
        'photograph': 1,
        'illustration': 2,
        'map': 3,
        'cartoon': 4,
        'headline': 5,
        'advertisement': 6,
    }
    if reverse:
        label_dict = {y: x for x, y in label_dict.items()}
    return label_dict[label]


def show_random_img_with_all_annotations(
    in_list: List,
    expected_list: List,
    path_to_photos: str,
    matplotlib_colours_dict: Dict,
    confidence_level: float = 0.2,
    pages: int = 5,
    jupyter=True,
) -> None:
    """
    Show the number (pages) of images with all the annotations in given confidence level of the prediction
    """
    prev = []
    for i in range(pages):
        random_img = random.randint(0, len(in_list)-1)
        if random_img in prev:
            i -= 1
        prev.append(random_img)
        file_name = in_list[random_img]

        with cbook.get_sample_data(path_to_photos + file_name) as image_file:
            image = plt.imread(image_file)

        if jupyter:
            fig, ax = plt.subplots(figsize=(15, 10))
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
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
                rect = Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=1,
                    edgecolor=matplotlib_colours_dict[annotation[0]],
                    facecolor="none",
                )
                ax.add_patch(rect)
                t = ax.text(
                    x0,
                    y0-50 if jupyter else y0-140,
                    cat_name,
                    fontsize=8 if jupyter else 5,
                    color=matplotlib_colours_dict[annotation[0]],
                )
                t.set_bbox(dict(facecolor='black', alpha=0.8, edgecolor='black'))
        
        plt.title(file_name)
        plt.show()


def predict_one_img(
    model: torchvision.models.detection.FasterRCNN,
    dataloader: torch.utils.data.DataLoader,
    image_name: str,
    path_to_image: str,
) -> None:
    """
    Function to predict single image in predict script
    """
    device = torch.device("cpu")
    model.to(device)
    # evaluation
    with torch.no_grad():
        img_names_list, predicted_bboxes_list, predicted_labels_list, scores_list, new_sizes_list = [], [], [], [], []
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]
            for target in list(targets):
                img_names_list.append(
                    int([t.detach().numpy() for t in target["image_name"]][0])
                )
                new_sizes_list.append(
                    [
                        t.detach().numpy().tolist()
                        for t in target["new_image_size"]
                    ]
                )
            outputs = model(images)
            outputs = [
                {k: v.to(device) for k, v in t.items()} for t in outputs
            ]
            for output in outputs:
                predicted_labels_list.append(
                    [int(o.detach().numpy()) for o in output["labels"]]
                )
                predicted_bboxes_list.append(
                    [o.detach().numpy().tolist() for o in output["boxes"]]
                )
                scores_list.append(
                    [o.detach().numpy().tolist() for o in output["scores"]]
                )

        return parse_model_outcome([image_name], path_to_image, new_sizes_list, predicted_labels_list, scores_list, predicted_bboxes_list)


def parse_model_outcome(img_names_list: List[str], image_directory: str, new_sizes_list: List, predicted_labels_list: List[int], scores_list: List[float], predicted_bboxes_list: List[int]) -> List[str]:
    """
    Parse model outcome to list (list of lists where 1 element contains all annotations for 1 image)
    """
    out_list = []
    for i in range(len(img_names_list)):
        img_width, img_height = get_image_size.get_image_size(f'{image_directory}{img_names_list[i]}')
        out_str = ''
        for ii in range(len(predicted_labels_list[i])):
            scaler_width = np.float(img_width)/np.float(new_sizes_list[i][0][0])
            scaler_height = np.float(img_height)/np.float(new_sizes_list[i][0][1])
            label = target_encoder(int(predicted_labels_list[i][ii]), reverse=True)
            x0 = str(int(round(predicted_bboxes_list[i][ii][0],0)*scaler_width))
            y0 = str(int(round(predicted_bboxes_list[i][ii][1],0)*scaler_height))
            x1 = str(int(round(predicted_bboxes_list[i][ii][2],0)*scaler_width))
            y1 = str(int(round(predicted_bboxes_list[i][ii][3],0)*scaler_height))

            out_str = f'{out_str}{label}:{x0},{y0},{x1},{y1}:{scores_list[i][ii]} '

        out_str = out_str.strip(" ")
        out_list.append(out_str)

    return out_list


def get_input_statistics(expected_list: List[str]) -> Tuple:
    """
    Get class size statistics of final model input list
    """
    count_series = pd.Series(
        [[
            obj.split(':')[0] for obj in obs.split(' ')
            ] for obs in expected_list]
        ).explode().reset_index(drop=True).value_counts()

    count_sum = sum(count_series)

    return count_series.to_string(), count_sum


def remove_if_missed_annotations(coco_file: Dict) -> List[int]:
    """
    Check that each image stored in coco-like dictionary has annotations and remove images without them.
    """
    image_ids_1 = [coco_file['images'][i]['id'] for i in range(len(coco_file['images']))]
    image_ids_2 = [coco_file['annotations'][i]['image_id'] for i in range(len(coco_file['annotations']))]

    missing_images_id_list = list(set(image_ids_1) - set(image_ids_2))

    to_remove = [i for i in range(len(coco_file['images'])) if coco_file['images'][i]['id'] in missing_images_id_list]

    for index in to_remove:
        del coco_file['images'][index]

    return coco_file


def input_transformer(coco_file: Dict, path_to_photos_dir: str) -> Tuple:
    """
    Save preprocessed annotations and image names to lists.

    Returns:
        Tuple: (image names list, image annoations list)
    """
    in_list, expected_list = [], []
    for i in range(len(coco_file['images'])):
        img_string = ' '
        img_name = coco_file['images'][i]['file_name']
        in_list.append(img_name)
        for ii in range(len(coco_file['annotations'])):
            if int(coco_file['images'][i]['id']) == int(coco_file['annotations'][ii]['image_id']):
                img_width, img_height = get_image_size.get_image_size(
                    path_to_photos_dir + coco_file['images'][i]['file_name']
                )
                cat = coco_file['categories'][int(coco_file['annotations'][ii]['category_id'])]['name'].lower()
                if cat in ['comics/cartoon', 'editorial cartoon']:
                    cat = 'cartoon'
                x0 = coco_file['annotations'][ii]['bbox'][0]
                y0 = coco_file['annotations'][ii]['bbox'][1]
                x1 = x0 + coco_file['annotations'][ii]['bbox'][2]
                y1 = y0 + coco_file['annotations'][ii]['bbox'][3]
                if x0 < 0:
                    x0 = 1
                if y0 < 0:
                    y0 = 1
                if x0 > img_width:
                    continue
                if y0 > img_height:
                    continue
                if x1 < 0:
                    continue
                if y1 < 0:
                    continue
                if x1 > img_width:
                    x1 = img_width-1
                if y1 > img_height:
                    y1 = img_height-1
                img_string = f"{img_string}{cat.lower()}:{x0},{y0},{x1},{y1} "

        img_string = img_string.strip()
        if img_string == '':
            del in_list[-1]
        else:
            expected_list.append(img_string)

    return in_list, expected_list


def remove_coco_elem_if_in_list(coco_file: Dict, key_name: str, list_with_names_to_remove: List[str], id_key: str) -> Dict:
    """
    Remove all names from the declared list in the coco-like annotation dictionary for a given key
    """
    indexes_to_remove = [i for i in range(len(coco_file[key_name])) if str(coco_file[key_name][i][id_key]) in list_with_names_to_remove]
    for index in sorted(indexes_to_remove, reverse=True):
        del coco_file[key_name][int(index)]

    return coco_file


def calcMD5Hash(name: str, split: float) -> bool:
    """
    Hashing function needed for train test split
    """
    return int(hashlib.md5(name.encode('Utf-8')).digest()[-1]) <= split * 256


def rescale_annotations(coco_metadata: Dict, directory: str) -> Dict:
    """
    Rescale annotations based on the previous and actual image size.
    It's needed because scraped images are of higher resolution.
    """
    for i in range(len(coco_metadata['images'])):
        iterable_path_i = coco_metadata['images'][i]
        file_name = iterable_path_i['file_name']
        scraped_photo_width, scraped_photo_height = get_image_size.get_image_size(f'{directory}/scraped_photos/{file_name}')

        scaler_height = scraped_photo_height/iterable_path_i['height']
        scaler_width = scraped_photo_width/iterable_path_i['width']

        iterable_path_i['height'], iterable_path_i['width'] = scraped_photo_height, scraped_photo_width

        for ii in range(len(coco_metadata['annotations'])):
            if int(coco_metadata['annotations'][ii]['image_id']) == int(iterable_path_i['id']):
                iterable_path_ii = coco_metadata['annotations'][ii]
                # bbox in coco should be [x,y,width,height]
                iterable_path_ii['bbox'][0] = int(iterable_path_ii['bbox'][0]*scaler_width)
                iterable_path_ii['bbox'][2] = int(iterable_path_ii['bbox'][2]*scaler_width)
                iterable_path_ii['bbox'][1] = int(iterable_path_ii['bbox'][1]*scaler_height)
                iterable_path_ii['bbox'][3] = int(iterable_path_ii['bbox'][3]*scaler_height)

                iterable_path_ii['area'] = iterable_path_ii['bbox'][2]*iterable_path_ii['bbox'][3]

    return coco_metadata


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
    cpu_device = torch.device("cpu")
    if cuda_statement:
        torch.multiprocessing.set_sharing_strategy('file_system')
    # predict
    with torch.no_grad():
        f_out, f_tar = [], []
        for images, targets in tqdm(dataloader, desc='Evaluation'):
            if cuda_statement:
                images = [img.to(device) for img in images]
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


def prepare_data_for_ap(output_list: List, target_list: List) -> Tuple[List[np.ndarray]]:
    """
    Prepare data into a format adequate for AP calculation supporting both: raw model output and model output stored in tsv files
    """
    p_output_list  = []
    for one_image_annotations in output_list:
        temp_pred = []
        for annotation in one_image_annotations.split(' '):
            boxes = [int(box) for box in annotation.split(':')[1].split(',')]
            label = int(target_encoder(annotation.split(':')[0], reverse=False)) - 1
            score = float(annotation.split(':')[2])
            temp_pred.append(boxes + [label, score])
        
        p_output_list.append(np.array(temp_pred))
    
    p_ground_truth_list = []
    for one_image_annotations in target_list:    
        temp_gt = []
        for annotation in one_image_annotations.split(' '):
            boxes = [int(box) for box in annotation.split(':')[1].split(',')]
            label = int(target_encoder(annotation.split(':')[0], reverse=False)) - 1
            difficult, crowd = 0, 0
            temp_gt.append(boxes + [label, difficult, crowd])
            
        p_ground_truth_list.append(np.array(temp_gt))

    return p_output_list, p_ground_truth_list
 

def calculate_map(
    prepared_pred_list: List[np.array],
    prepared_ground_truth_list: List[np.array],
    num_classes: int,
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
        "map_2d", async_mode=True, num_classes=6
    )

    for i in range(len(prepared_pred_list)):
        metric_fn.add(prepared_pred_list[i], prepared_ground_truth_list[i])

    metric = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )

    return {
        "photograph": round(metric[0.5][0]["ap"],4),
        "illustration": round(metric[0.5][1]["ap"],4),
        "map": round(metric[0.5][2]["ap"],4),
        "cartoon": round(metric[0.5][3]["ap"],4),
        "headline": round(metric[0.5][4]["ap"],4),
        "advertisement": round(metric[0.5][5]["ap"],4),
        "mean": round(metric["mAP"],4),
    }


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
