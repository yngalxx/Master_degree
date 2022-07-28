import json
import logging
import math
import sys
import time
import warnings
from contextlib import redirect_stdout

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from functions_catalogue import (calculate_map, predict_eval_set,
                                 prepare_data_for_ap)

# warnings
warnings.filterwarnings("ignore")


def train_model(
    pre_treined_model: torchvision.models.detection.FasterRCNN,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    val_dataloader: torch.utils.data.DataLoader = None,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None,
    gpu: bool = True,
) -> torchvision.models.detection.FasterRCNN:
    training_start_time = time.time()
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    logging.info(f"Cuda available: {cuda_statement}")
    if cuda_statement and gpu:
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    logging.info(f"Current device: {device}\n")
    # move model to the right device
    pre_treined_model.to(device)
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_start_time = time.time()
        pre_treined_model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_dataloader):
            if cuda_statement and gpu:
                images = [image.to(device) for image in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            loss_dict = pre_treined_model(images, targets)
            losses = sum(loss_dict.values())
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                logging.error(
                    f"Loss is {loss_value}, stopping training", exc_info=True
                )
                sys.exit(1)
            # calculate gradients
            losses.backward()
            # update weights
            optimizer.step()
            # calculate loss
            train_loss += loss_value

        logging.info(
            f"[epoch {epoch + 1}] Epoch train time:"
            f" {round(time.time()-epoch_start_time,2)} sec. | train loss ="
            f" {train_loss / len(train_dataloader)}"
        )

        # update the learning rate
        if lr_scheduler:
            lr_scheduler.step()

        # evaluate on the validation dataset
        if val_dataloader:
            logging.info("Calculating mAP metric on validation set")
            out, targ = predict_eval_set(
                dataloader=val_dataloader,
                model=pre_treined_model,
                device=device,
            )
            prep_pred, prepr_gt = prepare_data_for_ap(out, targ)
            eval_metrics = calculate_map(prep_pred, prepr_gt)
            with redirect_stdout(logging):
                print(json.dumps(eval_metrics, indent=4))
                print(
                    "Metric"
                    f' results:\n{pd.DataFrame.from_dict(eval_metrics, orient="index", columns=["AP"]).to_string()}'
                )

    logging.info(
        "Model training completed, runtime:"
        f" {round(time.time()-training_start_time,2)} sec."
    )

    return pre_treined_model
