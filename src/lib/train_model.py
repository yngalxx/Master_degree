import logging
import math
import time
import warnings

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from lib.metric import calculate_map, prepare_eval_out_for_ap
from lib.predict import predict_eval_set

# warnings
warnings.filterwarnings("ignore")


def train_model(
    model: torchvision.models.detection.FasterRCNN,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    num_classes: int,
    val_map_threshold: float,
    force_save_model: bool,
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
                prep_pred, prepr_gt, num_classes=num_classes
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
