import warnings
from typing import Dict, List, Tuple, Union

import torch
import torchvision
from tqdm import tqdm

from lib.postprocessing import parse_model_outcome

# warnings
warnings.filterwarnings("ignore")


def predict_eval_set(
    dataloader: torch.utils.data.DataLoader,
    model: torchvision.models.detection.FasterRCNN,
    device: Union[torch.device, None],
) -> Tuple[List[Dict]]:
    """
    Make prediction on dataloader and return tuple with two lists containing
    output dictionaries
    """
    # settings
    cuda_statement = torch.cuda.is_available()
    cpu_device = torch.device("cpu")
    if cuda_statement:
        torch.multiprocessing.set_sharing_strategy("file_system")
    # predict
    with torch.no_grad():
        f_out, f_tar = [], []
        for images, targets in tqdm(dataloader, desc="Evaluation"):
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


def predict_one_img(
    model: torchvision.models.detection.FasterRCNN,
    dataloader: torch.utils.data.DataLoader,
    image_name: str,
    path_to_image: str,
    class_coding_dict: Dict,
) -> None:
    """
    Function to predict single image in predict script
    """
    device = torch.device("cpu")
    model.to(device)
    # evaluation
    with torch.no_grad():
        (
            img_names_list,
            predicted_bboxes_list,
            predicted_labels_list,
            scores_list,
            new_sizes_list,
        ) = ([], [], [], [], [])
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

        return parse_model_outcome(
            [image_name],
            path_to_image,
            new_sizes_list,
            predicted_labels_list,
            scores_list,
            predicted_bboxes_list,
            class_coding_dict,
        )
