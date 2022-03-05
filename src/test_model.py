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


def model_predict(model, save_path, test_dataloader, gpu=True):
    start_time = datetime.datetime.now()
    print(f'Start time: {start_time} \n')
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    print(f'Cuda available: {torch.cuda.is_available()}')
    if cuda_statement == True:
        device = torch.device(torch.cuda.current_device())
    if gpu==False:
        device = 'cpu'
    print(f'Current device: {device}\n')
    # move model to the right device
    model.to(device)
    print('###  Evaluaiting  ###')
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        image_id_list, img_name_list, predicted_box_list, predicted_label_list, old_size_list, new_size_list = [], [], [], [], [], []
        for images, targets in tqdm(test_dataloader):
            if cuda_statement == True:
                images = list(img.to(device) for img in images)
                torch.cuda.synchronize()
            else:
                images = list(img.to(cpu_device) for img in images)
            targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
            for target in list(targets):
                image_id_list.append(target['image_id'].item())
                img_name_list.append([int(t.detach().numpy()) for t in target['image_name']])
                new_size_list.append([t.detach().numpy().tolist() for t in target['new_image_size']])
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            for output in outputs:
                predicted_label_list.append([int(o.detach().numpy()) for o in output['labels']])
                predicted_box_list.append([o.detach().numpy().tolist() for o in output['boxes']])
        output_df = pd.DataFrame()
        output_df['image_id'] = image_id_list
        output_df['image_name'] = img_name_list
        output_df['predicted_labels'] = predicted_label_list
        output_df['predicted_boxes'] = predicted_box_list
        output_df['new_image_size'] = new_size_list
        output_df = output_df.sort_values('image_name').reset_index(drop=True, inplace=False)
        output_df.to_csv(save_path)
 
    print(f'\n####### JOB FINISHED #######\n\n')

    return model
