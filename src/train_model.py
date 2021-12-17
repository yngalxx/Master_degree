import sys, datetime
import torch
import math


def train_model(model, optimizer, train_dataloader, epochs, val_dataloader=None, lr_scheduler=None):
    start_time = datetime.datetime.now()
    print(f'Start time: {start_time} \n')
    # switch to gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print(f"ERROR: Loss is {loss_value}, stopping training")
                sys.exit(1)
            # calculate gradients
            losses.backward()
            # update weights
            optimizer.step()
            # calculate loss
            train_loss += loss_value

        train_print = f'{datetime.datetime.now()} - epoch {epoch + 1}: train_loss = {train_loss / len(train_dataloader)}'

        if val_dataloader:
            model.eval()
            val_loss = 0.0
            for images, targets in val_dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                if not math.isfinite(loss_value):
                    print(f"ERROR: Loss is {loss_value}, stopping training")
                    sys.exit(1)
                val_loss += loss_value

            print(train_print + f' val_loss = {val_loss / len(val_dataloader)}')
        else:
            print(train_print)

        if lr_scheduler:
            lr_scheduler.step()

    print(f'\nModel training completed, runtime: {datetime.datetime.now() - start_time}')

    return model
