import sys
import datetime
import torch
import math
from tqdm import tqdm
# from functions import iou_list
# from sklearn.metrics import f1_score
sys.path.append('/home/wmi/adrozdz/vision/references/detection/')
from engine import train_one_epoch, evaluate # source repository: https://github.com/pytorch/vision/tree/main/references/detection from tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

def train_model(model, optimizer, train_dataloader, epochs, num_classes, gpu=True, save_path=False, val_dataloader=None, lr_scheduler=None):
    start_time = datetime.datetime.now()
    print(f'Start time: {start_time} \n')
    # switch to gpu if available
    if gpu:
        cuda_statement = torch.cuda.is_available()
    print(f'Cuda available: {torch.cuda.is_available()}')
    if gpu == True & cuda_statement == True:
        print(f'Current device: {torch.cuda.current_device()}')
        device = torch.device(torch.cuda.current_device())
        # move model to the righ
        model.to(device)
    else:
        device = 'cpu'

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        if lr_scheduler:
            lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_dataloader, device=device)

    # for epoch in range(epochs):        
    #     model.train()
    #     train_loss = 0.0
    #     for images, targets in tqdm(train_dataloader):
    #         if cuda_statement == True:
    #            images = list(image.to(device) for image in images)
    #            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #         # clear the gradients
    #         optimizer.zero_grad()
    #         # forward pass
    #         loss_dict = model(images, targets)
    #         losses = sum(loss for loss in loss_dict.values())
    #         loss_value = losses.item()
    #         if not math.isfinite(loss_value):
    #            print(f"ERROR: Loss is {loss_value}, stopping training")
    #            sys.exit(1)
    #         # calculate gradients
    #         losses.backward()
    #         # update weights
    #         optimizer.step()
    #         # calculate loss
    #         train_loss += loss_value

    #     if val_dataloader:
    #         model.eval()
    #         val_loss = 0.0
    #         with torch.no_grad():
    #            boxes_true, boxes_pred = [], []
    #            lables_true, labels_pred = [], []
    #            for images, targets in tqdm(val_dataloader):
    #                 if cuda_statement == True:
    #                     images = list(image.to(device) for image in images)
    #                     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #                 output = model(images) # to zwraca dziwną liste pustych tensorów w dictcie, takie: [{'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0')}, {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0')}, {'boxes': tensor([], device='cuda:0', size=(0, 4)), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0')}, {'boxes': tensor([], device='cuda:0', size=(0, 4)), ...]
    #                 print(output[0])
    #                 print(targets)
    #                 boxes_true.append(targets['boxes'].tolist()), boxes_pred.append(output['boxes'].tolist())
    #                 lables_true.append(targets['labels'].tolist()), labels_pred.append(output['labels'].tolist())
    #         mean_iou_list = iou_list(boxes_true, boxes_pred)
    #         f1_per_class = f1_score(lables_true, labels_pred, average=None)
    #         print(f'### {datetime.datetime.now()} ### [epoch {epoch + 1}]: train_time = {datetime.datetime.now()-start_time} | train_loss = {train_loss / len(train_dataloader)} | val_iou = {sum(mean_iou_list)/len(mean_iou_list)} | val_f1_per_class = {f1_per_class}')
    #     else:
    #         print(f'### {datetime.datetime.now()} ### [epoch {epoch + 1}]: train_time = {datetime.datetime.now()-start_time} | train_loss = {train_loss / len(train_dataloader)}')

    #     if lr_scheduler:
    #        lr_scheduler.step()

    if save_path:
        torch.save(model, save_path)

    print(f'\nModel training completed, runtime: {datetime.datetime.now() - start_time}')

    return model
