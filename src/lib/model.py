import logging
import math
import os
import sys
import time
import warnings
from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from lib.metric import (calculate_map, prepare_data_for_ap,
                        prepare_eval_out_for_ap)
from lib.postprocessing import parse_model_outcome
from lib.preprocessing import get_statistics
from lib.save_load_data import (dump_json, from_tsv_to_list,
                                save_list_to_tsv_file)

# warnings
warnings.filterwarnings("ignore")


def initalize_model(
    pretrained: bool, trainable_backbone_layers: int, num_classes: int
) -> torchvision.models.detection.FasterRCNN:
    # pre-trained resnet50 model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # replace the pre-trained head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes=num_classes
        )
    )

    return model


def initialize_optimizer(
    torch_model_parameters: Iterator,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler: bool,
    lr_step_size: int,
    lr_gamma: float,
) -> Tuple:
    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, torch_model_parameters),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # learning rate scheduler decreases the learning rate by 'gamma' every 'step_size'
    if lr_scheduler:
        lrs = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
    else:
        lrs = None

    return optimizer, lrs


def train_model(
    model: torchvision.models.detection.FasterRCNN,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    num_classes: int,
    class_names: List,
    val_map_threshold: float,
    force_save_model: bool,
    val_dataloader: torch.utils.data.DataLoader = None,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None,
    gpu: bool = True,
) -> Tuple:
    training_start_time = time.time()
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    logging.info(f"Cuda available: {cuda_statement}")
    if cuda_statement and gpu:
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    logging.info(f"Current device: {device}")
    # move model to the right device
    model.to(device)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(
            train_dataloader, desc=f"[Epoch {epoch + 1}]"
        ):
            if cuda_statement and gpu:
                images = [image.to(device) for image in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                logging.error(
                    f"Loss is {loss_value}, stopping training", exc_info=True
                )
                raise ValueError()
            # calculate gradients
            losses.backward()
            # update weights
            optimizer.step()
            # calculate loss
            train_loss += loss_value

        logging.info(
            f"[Epoch {epoch + 1}] Epoch train time:"
            f" {round(time.time()-epoch_start_time,2)} sec. | train loss ="
            f" {round(train_loss / len(train_dataloader),4)}"
        )

        # update the learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # evaluate on the validation dataset
        if val_dataloader:
            model.eval()
            logging.info("Evaluating validation set")
            out, targ = predict_eval_set(
                dataloader=val_dataloader,
                model=model,
                device=device,
            )
            prep_pred, prepr_gt = prepare_eval_out_for_ap(out, targ)
            logging.info("Calculating mAP metric on validation set")
            eval_metrics = calculate_map(
                prep_pred,
                prepr_gt,
                num_classes=num_classes,
                class_names=class_names,
            )
            eval_df = pd.DataFrame.from_dict(
                eval_metrics, orient="index", columns=["AP"]
            )
            logging.info(f"Metric results:\n{eval_df.to_string()}")

            if eval_df["AP"]["mean"] >= val_map_threshold:
                logging.info(
                    f"Training was stopped after epoch number {epoch} due to"
                    " exceeding threshold set for evaluation metric value"
                    f" ({eval_df['AP']['mean']:.4f}>={val_map_threshold:.4f})."
                )
                force_save_model = True
                break
        else:
            eval_df = None

    logging.info(
        "Model training completed, runtime:"
        f" {round(time.time()-training_start_time,2)} sec."
    )

    return model, eval_df, force_save_model


def evaluate_model(
    model: torchvision.models.detection.FasterRCNN,
    dataloader: torch.utils.data.DataLoader,
    main_dir: str,
    save_path: str,
    expected_list_exists: bool,
    num_classes: int,
    class_names: List[str],
    class_coding_dict: Dict,
    gpu: bool = False,
) -> None:
    evaluation_start_time = time.time()
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    logging.info(f"Cuda available: {cuda_statement}")
    device = (
        torch.device(torch.cuda.current_device())
        if cuda_statement and gpu
        else torch.device("cpu")
    )

    logging.info(f"Current device: {device}")
    # move model to the right device
    model.to(device)
    cpu_device = torch.device("cpu")
    # switch model stage to evaluation
    model.eval()
    # evaluation
    with torch.no_grad():
        (
            img_names_list,
            predicted_bboxes_list,
            predicted_labels_list,
            scores_list,
            new_sizes_list,
        ) = ([], [], [], [], [])
        if expected_list_exists:
            model_output, ground_truth = [], []
        for images, targets in tqdm(dataloader, desc="Evaluation"):
            if cuda_statement and gpu:
                images = [img.to(device) for img in images]
                torch.cuda.synchronize()
            else:
                images = [img.to(cpu_device) for img in images]
            targets = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in targets
            ]
            if expected_list_exists:
                ground_truth.append(targets)
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
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            if expected_list_exists:
                model_output.append(outputs)
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

    # save outputs
    logging.info("Saving model output")
    new_in_list = [f"{img}.jpg" for img in img_names_list]
    out_list = parse_model_outcome(
        new_in_list,
        f"{main_dir}/scraped_photos/",
        new_sizes_list,
        predicted_labels_list,
        scores_list,
        predicted_bboxes_list,
        class_coding_dict,
    )

    save_list_to_tsv_file(f"{main_dir}/data/{save_path}/out.tsv", out_list)
    save_list_to_tsv_file(f"{main_dir}/data/{save_path}/in.tsv", new_in_list)

    stats, sum = get_statistics(out_list, index=0)
    logging.info(f"Number of annotations predicted: {sum}")
    logging.info(f"Predicted labels statistics:\n{stats}")

    # mAP metric calculations
    if expected_list_exists:
        logging.info("Calculating mAP metric")
        out = from_tsv_to_list(f"{main_dir}/data/{save_path}/out.tsv")
        try:
            expected = from_tsv_to_list(
                f"{main_dir}/data/{save_path}/expected.tsv"
            )
        except FileNotFoundError as err:
            logging.exception(
                f"File 'expected.tsv' not found, value of metric cannot be calculated\nError: {err}")
            sys.exit(1)

        model_output, ground_truth = prepare_data_for_ap(
            out, expected, class_coding_dict
        )
        map_df = pd.DataFrame.from_dict(
            calculate_map(
                model_output,
                ground_truth,
                num_classes=num_classes,
                class_names=class_names,
            ),
            orient="index",
            columns=["AP"],
        )
        logging.info(f"Metric results:\n{map_df.to_string()}")

    logging.info(
        "Model evaluation completed, runtime:"
        f" {round(time.time()-evaluation_start_time,2)} sec."
    )


def create_model_config(
    channel: int,
    num_classes: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    rescale: bool,
    shuffle: bool,
    weight_decay: float,
    lr_scheduler: float,
    lr_step_size: int,
    lr_gamma: float,
    trainable_backbone_layers: int,
    num_workers: int,
    gpu: bool,
    bbox_format: str,
    pretrained: bool,
) -> Dict:
    return {
        "channel": channel,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "rescale": rescale,
        "shuffle": shuffle,
        "weight_decay": weight_decay,
        "lr_scheduler": lr_scheduler,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
        "trainable_backbone_layers": trainable_backbone_layers,
        "num_workers": num_workers,
        "gpu": gpu,
        "bbox_format": bbox_format,
        "pretrained": pretrained,
    }


def check_model_and_save(
    model_path: str,
    evaluation_df: pd.DataFrame,
    val_dataloader: torch.utils.data.DataLoader,
    force_save_model: bool,
    model_config: Dict,
    trained_model_state_dict: Dict,
) -> bool:
    model_metric_path = f"{model_path}model_eval_metric.csv"
    if val_dataloader and not force_save_model:
        if not os.path.exists(model_metric_path):
            logging.info(
                "Previous model validation results not found, model"
                " will be saved"
            )
            force_save_model = True
        else:
            prev_evaluation_df = pd.read_csv(model_metric_path, index_col=0)
            prev_results_map, current_results_map = (
                prev_evaluation_df["AP"]["mean"],
                evaluation_df["AP"]["mean"],
            )
            if current_results_map >= prev_results_map:
                logging.info(
                    "Model validation results (mAP ="
                    f" {current_results_map:.4f}) are better than"
                    f" previous ones (mAP = {prev_results_map:.4f}),"
                    " previous model will be overridden"
                )
                force_save_model = True
            else:
                logging.info(
                    "Model validation results (mAP ="
                    f" {current_results_map:.4f}) are worse than"
                    f" previous ones (mAP = {prev_results_map:.4f}),"
                    " model will not be saved and evaluation phase"
                    " will be skipped"
                )
                force_save_model = False

    if force_save_model:
        config_dir_name = model_path.split("/")[-1]
        dump_json(f"{model_path}model_config.json", model_config)
        logging.info(
            f'Model config json saved in "{config_dir_name}" directory'
        )
        torch.save(trained_model_state_dict, f"{model_path}model.pth")
        logging.info(f'Model saved in "{config_dir_name}" directory')

        if val_dataloader:
            evaluation_df.to_csv(model_metric_path)
            logging.info(
                "Model validation results saved in"
                f' "{config_dir_name}" directory'
            )

        if not val_dataloader and os.path.exists(model_metric_path):
            os.remove(model_metric_path)

        evaluate = True
    else:
        evaluate = False

    return evaluate


def load_model_state_dict(
    gpu: bool,
    init_model: torchvision.models.detection.FasterRCNN,
    config_dir_path: str,
) -> torchvision.models.detection.FasterRCNN:
    if torch.cuda.is_available() and gpu:
        map_location = torch.device(torch.cuda.current_device())
    else:
        map_location = torch.device("cpu")

    init_model.load_state_dict(
        torch.load(
            f"{config_dir_path}model.pth",
            map_location=map_location,
        ),
        strict=True,
    )
    return init_model


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
    # switch model stage to evaluation
    model.eval()
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
