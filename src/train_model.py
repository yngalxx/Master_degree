import sys
import datetime
import math
from tqdm import tqdm
from functions import save_list_to_txt_file
from sklearn.metrics import f1_score
import torch
import pandas as pd
import torch.optim as optim
import torchvision.transforms as T
from newspapersdataset import NewspapersDataset
import csv
import warnings
from torch.utils.data import DataLoader
import pathlib
sys.path.append("/".join(str(pathlib.Path(__file__).parent.resolve()).split('/')[:-2])+'/vision/references/detection/')
from engine import evaluate # source repository: https://github.com/pytorch/vision/tree/main/references/detection from tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# warnings
warnings.filterwarnings("ignore")


def train_model(model, optimizer, train_dataloader, epochs, gpu=True, save_path=False, val_dataloader=None, lr_scheduler=None):
    start_time = datetime.datetime.now()
    print(f'Start time: {start_time} \n')
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    print(f'Cuda available: {torch.cuda.is_available()}')
    if cuda_statement == True:
        device = torch.device(torch.cuda.current_device())
    else:
        device = 'cpu'
    if gpu==False:
        device = 'cpu'
    print(f'Current device: {device}\n')
    # move model to the right device
    model.to(device)
    print('###  Training  ###')
    for epoch in range(epochs):    
        print(f'###  Epoch: {epoch+1}  ###')    
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_dataloader):
            if cuda_statement == True:
               images = list(image.to(device) for image in images)
               targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            loss_dict = model(images, targets)
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
        
        print(f'### {datetime.datetime.now()} ### [epoch {epoch + 1}]: train_time = {datetime.datetime.now()-start_time} | train_loss = {train_loss / len(train_dataloader)}')
        
        # update the learning rate
        if lr_scheduler:
            lr_scheduler.step()
        # evaluate on the validation dataset
        if val_dataloader:
            print(f'### Evaluation ###') 
            evaluate(model, val_dataloader, device=device)

    print(f'\n### Model training completed, runtime: {datetime.datetime.now() - start_time} ###')

    if save_path:
        torch.save(model, save_path+'saved_models/model.pth')

    print(f'\n####### JOB FINISHED #######\n\n')

    return model
