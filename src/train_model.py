import datetime
import math
import pathlib
import sys
import warnings

import torch
import torchvision
from tqdm import tqdm

sys.path.append(
    "/".join(str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2])
)
from vision.references.detection.engine import (
    evaluate,
)  # source repository: https://github.com/pytorch/vision/tree/main/references/detection (small fix was needed) from tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

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
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time} \n")
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    print(f"Cuda available: {torch.cuda.is_available()}")
    if cuda_statement == True:
        device = torch.device(torch.cuda.current_device())
    else:
        device = "cpu"
    if gpu == False:
        device = "cpu"
    print(f"Current device: {device}\n")
    # move model to the right device
    pre_treined_model.to(device)
    print("###  Training  ###")
    for epoch in range(epochs):
        print(f"###  Epoch: {epoch+1}  ###")
        pre_treined_model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_dataloader):
            if cuda_statement == True:
                images = list(image.to(device) for image in images)
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            loss_dict = pre_treined_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print(f"\nERROR: Loss is {loss_value}, stopping training")
                sys.exit(1)
            # calculate gradients
            losses.backward()
            # update weights
            optimizer.step()
            # calculate loss
            train_loss += loss_value

        print(
            f"### {datetime.datetime.now()} ### [epoch {epoch + 1}]:"
            f" train_time = {datetime.datetime.now()-start_time} | train_loss"
            f" = {train_loss / len(train_dataloader)}"
        )

        # update the learning rate
        if lr_scheduler:
            lr_scheduler.step()
        # evaluate on the validation dataset
        if val_dataloader:
            print(f"### Evaluation ###")
            evaluate(pre_treined_model, val_dataloader, device=device)

    print(
        "\n### Model training completed, runtime:"
        f" {datetime.datetime.now() - start_time} ###"
    )

    print(f"\n####### JOB FINISHED #######\n\n")

    return pre_treined_model
