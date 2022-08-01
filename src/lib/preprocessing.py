import hashlib
from typing import Dict, List, Tuple

import imagesize
import pandas as pd


def get_statistics(expected_list: List[str], index: int = 0) -> Tuple:
    """
    Get class size statistics of final model input list
    """
    count_series = (
        pd.Series(
            [
                [obj.split(":")[index] for obj in obs.split(" ")]
                for obs in expected_list
            ]
        )
        .explode()
        .reset_index(drop=True)
        .value_counts()
    )
    count_df = pd.DataFrame(count_series, columns=["count"])

    count_sum = sum(count_series)

    return count_df.to_string(), count_sum


def remove_if_missed_annotations(coco_file: Dict) -> List[int]:
    """
    Check that each image stored in coco-like dictionary has annotations and
    remove images without them.
    """
    image_ids_1 = [
        coco_file["images"][i]["id"] for i in range(len(coco_file["images"]))
    ]
    image_ids_2 = [
        coco_file["annotations"][i]["image_id"]
        for i in range(len(coco_file["annotations"]))
    ]

    missing_images_id_list = list(set(image_ids_1) - set(image_ids_2))

    to_remove = [
        i
        for i in range(len(coco_file["images"]))
        if coco_file["images"][i]["id"] in missing_images_id_list
    ]

    for index in to_remove:
        del coco_file["images"][index]

    return coco_file


def input_transformer(
    coco_file: Dict, path_to_photos_dir: str, train_val_set: bool
) -> Tuple:
    """
    Save preprocessed annotations and image names to lists.

    Returns:
        Tuple: (image names list, image annoations list)
    """
    in_list = []
    expected_list = [] if train_val_set else None
    for i in range(len(coco_file["images"])):
        img_string = " "
        img_name = coco_file["images"][i]["file_name"]
        in_list.append(img_name)
        if train_val_set:
            for ii in range(len(coco_file["annotations"])):
                if int(coco_file["images"][i]["id"]) == int(
                    coco_file["annotations"][ii]["image_id"]
                ):
                    img_width, img_height = imagesize.get(
                        path_to_photos_dir
                        + coco_file["images"][i]["file_name"]
                    )
                    cat = coco_file["categories"][
                        int(coco_file["annotations"][ii]["category_id"])
                    ]["name"].lower()
                    if cat in ["comics/cartoon", "editorial cartoon"]:
                        cat = "cartoon"
                    x0 = coco_file["annotations"][ii]["bbox"][0]
                    y0 = coco_file["annotations"][ii]["bbox"][1]
                    x1 = x0 + coco_file["annotations"][ii]["bbox"][2]
                    y1 = y0 + coco_file["annotations"][ii]["bbox"][3]
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
                        x1 = img_width - 1
                    if y1 > img_height:
                        y1 = img_height - 1
                    img_string = (
                        f"{img_string}{cat.lower()}:{x0},{y0},{x1},{y1} "
                    )

            img_string = img_string.strip()
            if img_string == "":
                del in_list[-1]
            else:
                expected_list.append(img_string)

    return in_list, expected_list


def remove_coco_elem_if_in_list(
    coco_file: Dict,
    key_name: str,
    list_with_names_to_remove: List[str],
    id_key: str,
) -> Dict:
    """
    Remove all names from the declared list in the coco-like annotation
    dictionary for a given key
    """
    indexes_to_remove = [
        i
        for i in range(len(coco_file[key_name]))
        if str(coco_file[key_name][i][id_key]) in list_with_names_to_remove
    ]
    for index in sorted(indexes_to_remove, reverse=True):
        del coco_file[key_name][int(index)]

    return coco_file


def calcMD5Hash(name: str, split: float) -> bool:
    """
    Hashing function needed for train test split
    """
    return int(hashlib.md5(name.encode("Utf-8")).digest()[-1]) <= split * 256


def rescale_annotations(coco_metadata: Dict, directory: str) -> Dict:
    """
    Rescale annotations based on the previous and actual image size.
    It's needed because scraped images are of higher resolution.
    """
    for i in range(len(coco_metadata["images"])):
        iterable_path_i = coco_metadata["images"][i]
        file_name = iterable_path_i["file_name"]
        (
            scraped_photo_width,
            scraped_photo_height,
        ) = imagesize.get(f"{directory}/scraped_photos/{file_name}")

        scaler_height = scraped_photo_height / iterable_path_i["height"]
        scaler_width = scraped_photo_width / iterable_path_i["width"]

        iterable_path_i["height"], iterable_path_i["width"] = (
            scraped_photo_height,
            scraped_photo_width,
        )

        for ii in range(len(coco_metadata["annotations"])):
            if int(coco_metadata["annotations"][ii]["image_id"]) == int(
                iterable_path_i["id"]
            ):
                iterable_path_ii = coco_metadata["annotations"][ii]
                # bbox in coco should be [x,y,width,height]
                iterable_path_ii["bbox"][0] = int(
                    iterable_path_ii["bbox"][0] * scaler_width
                )
                iterable_path_ii["bbox"][2] = int(
                    iterable_path_ii["bbox"][2] * scaler_width
                )
                iterable_path_ii["bbox"][1] = int(
                    iterable_path_ii["bbox"][1] * scaler_height
                )
                iterable_path_ii["bbox"][3] = int(
                    iterable_path_ii["bbox"][3] * scaler_height
                )

                iterable_path_ii["area"] = (
                    iterable_path_ii["bbox"][2] * iterable_path_ii["bbox"][3]
                )

    return coco_metadata
