import random
from typing import Dict, List

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_random_img_with_all_annotations(
    in_list: List,
    expected_list: List,
    path_to_photos: str,
    matplotlib_colours_dict: Dict,
    confidence_level: float = 0.2,
    pages: int = 5,
) -> None:
    """
    Show the number (pages) of images with all the annotations in given confidence
    level of the prediction
    """
    prev = []
    for i in range(pages):
        random_img = random.randint(0, len(in_list) - 1)
        if random_img in prev:
            i -= 1
        prev.append(random_img)
        file_name = in_list[random_img]

        with cbook.get_sample_data(path_to_photos + file_name) as image_file:
            image = plt.imread(image_file)

        _, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(image, cmap="gray")

        if expected_list[random_img] != "":
            annotations = expected_list[random_img].split(" ")
            for annotation in annotations:
                annotation_temp = annotation.split(":")
                score = annotation_temp[2]
                if confidence_level > float(score):
                    continue
                bbox = annotation_temp[1].split(",")
                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                width, height = x1 - x0, y1 - y0
                cat_name = f"{annotation_temp[0]} {round(float(score)*100,2)}%"
                rect = Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=1,
                    edgecolor=matplotlib_colours_dict[annotation_temp[0]],
                    facecolor="none",
                )
                ax.add_patch(rect)
                t = ax.text(
                    x0,
                    y0,
                    cat_name,
                    fontsize=5,
                    color=matplotlib_colours_dict[annotation_temp[0]],
                )
                t.set_bbox(
                    dict(facecolor="black", alpha=0.8, edgecolor="black")
                )

        plt.title(file_name)
        plt.show()
