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
def warn(*args, **kwargs):
    pass

warnings.warn = warn


def train_model(model, optimizer, train_dataloader, epochs, gpu=True, save_path=False, val_dataloader=None,  test_dataloader=None, lr_scheduler=None):
    start_time = datetime.datetime.now()
    print(f'Start time: {start_time} \n')
    # switch to gpu if available
    if gpu:
        cuda_statement = torch.cuda.is_available()
    print(f'Cuda available: {torch.cuda.is_available()}')
    if gpu == True & cuda_statement == True:
        print(f'Current device: {torch.cuda.current_device()}\n')
        device = torch.device(torch.cuda.current_device())
    else:
        device = 'cpu'
        print(f'Device: {device}')
    # move model to the right device
    model.to(device)
    
    print('\n### Training ###')
    for epoch in range(epochs):    
        print(f'### Epoch: {epoch+1} ###')    
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
            evaluate(model, val_dataloader, device=device)

    print(f'\n### Model training completed, runtime: {datetime.datetime.now() - start_time} ###')

    if save_path:
        torch.save(model, save_path+'saved_models/model.pth')

    if test_dataloader:
        if save_path:
            print('\n### Testing ###')
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            cpu_device = torch.device("cpu")
            with torch.no_grad():
                image_id_list, img_name_list, predicted_box_list, true_box_list, true_label_list, predicted_label_list, old_size_list, new_size_list = [], [], [], [], [], [], [], []
                for images, targets in tqdm(test_dataloader):
                    images = list(image.to(cpu_device) for image in images)
                    targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
                    for target in list(targets):
                        image_id_list.append(target['image_id'].item())
                        img_name_list.append([int(t.detach().numpy()) for t in target['image_name']])
                        true_label_list.append([int(t.detach().numpy()) for t in target['labels']])
                        true_box_list.append([t.detach().numpy().tolist() for t in target['boxes']])
                        old_size_list.append([t.detach().numpy().tolist() for t in target['old_size']])
                        new_size_list.append([t.detach().numpy().tolist() for t in target['new_size']])
                    if cuda_statement == True:
                        images = list(img.to(device) for img in images)
                        torch.cuda.synchronize()
                    outputs = model(images)
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    for output in outputs:
                        predicted_label_list.append([int(o.detach().numpy()) for o in output['labels']])
                        predicted_box_list.append([o.detach().numpy().tolist() for o in output['boxes']])
                output_df = pd.DataFrame()
                output_df['image_id'] = image_id_list
                output_df['image_name'] = img_name_list
                output_df['true_labels'] = true_label_list
                output_df['true_boxes'] = true_box_list
                output_df['predicted_labels'] = predicted_label_list
                output_df['predicted_boxes'] = predicted_box_list
                output_df = output_df.sort_values('image_name').reset_index(drop=True, inplace=False)
                output_df.to_csv(save_path+'model_output/model_output.csv')
        else:
            print(f'\nERROR: Cannot process a test set without specifying the save_path parameter')

    print(f'\n####### JOB FINISHED #######\n')

    return model
