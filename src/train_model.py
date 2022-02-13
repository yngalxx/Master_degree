import sys
import datetime
import math
from tqdm import tqdm
from functions import iou_list, save_list_to_txt_file
from sklearn.metrics import f1_score
import torch
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.transforms as T
from newspapersdataset import NewspapersDataset
import csv
import warnings
from torch.utils.data import DataLoader
sys.path.append('/home/wmi/adrozdz/vision/references/detection/')
from engine import train_one_epoch, evaluate # source repository: https://github.com/pytorch/vision/tree/main/references/detection from tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
warnings.filterwarnings("ignore")


# class NewspaperNeuralNet(pl.LightningModule):
#     def __init__(self, model, parameters):
#         super(NewspaperNeuralNet, self).__init__()
#         self.model = model
#         self.parameters = parameters

#     def forward(self, x, *args, **kwargs):
#         return self.model(x)

#     def from_tsv_to_list(self, path, delimiter="\n"):
#         tsv_file = open(path)
#         read_tsv = csv.reader(tsv_file, delimiter=delimiter)
#         expected = list(read_tsv)

#         return [item for sublist in expected for item in sublist]

#     def prepare_data(self):
#         self.expected_train = self.from_tsv_to_list(self.parameters['annotations_dir'] + 'train/in.tsv')
#         self.in_train = self.from_tsv_to_list(self.parameters['annotations_dir'] + 'train/in.tsv')
#         self.expected_val = self.from_tsv_to_list(self.parameters['annotations_dir'] + 'dev-0/expected.tsv')
#         self.in_val = self.from_tsv_to_list(self.parameters['annotations_dir'] + 'dev-0/in.tsv')

#     def transform_data(self):
#         self.data_transform = T.Compose([
#             T.Grayscale(num_output_channels=self.parameters['channel']),
#             T.ToTensor(),
#             T.Normalize((0.5,), (0.5,)),
#         ])

#     def collate_fn(batch):
#         return tuple(zip(*batch))

#     def train_dataloader(self):
#         train_loader = DataLoader(
#             NewspapersDataset(
#                 img_dir=self.parameters['image_dir'],
#                 in_list=self.in_train,
#                 expected_list=self.expected_train,
#                 scale=self.parameters['rescale'],
#                 transforms=self.data_transform
#             ),
#             batch_size=self.parameters['batch_size'],
#             shuffle=self.parameters['shuffle'],
#             collate_fn=self.collate_fn,
#             num_workers = self.parameters['num_workers']
#         )

#         return train_loader

#     def val_dataloader(self):
#         val_loader = DataLoader(
#             NewspapersDataset(
#                 img_dir=self.parameters['image_dir'],
#                 in_list=self.in_val,
#                 expected_list=self.expected_val,
#                 scale=self.parameters['rescale'],
#                 transforms=self.data_transform
#             ),
#             batch_size=self.parameters['batch_size'],
#             shuffle=self.parameters['shuffle'],
#             collate_fn=self.collate_fn,
#             num_workers = self.parameters['num_workers']
#         )

#         return val_loader

#     def configure_optimizers(self):
#         optimizer = optim.Adam(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=self.parameters['learning_rate'],
#             weight_decay=self.parameters['weight_decay']
#         )

#         lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=self.parameters['step_size'],
#             gamma=self.parameters['gamma']
#         )

#         return [optimizer], [lr_scheduler]

#     def training_step(self, batch, batch_idx):
#         images, targets = batch
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         # separate losses
#         loss_dict = self.model(images, targets)
#         # total loss
#         losses = sum(loss for loss in loss_dict.values())

#         return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

#     def validation_step(self, batch, batch_idx):
#         images, targets = batch
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         outputs = self.model(images)
#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

#         return {}

#     def validation_epoch_end(self, outputs):
#         metric = 0
#         tensorboard_logs = {'main_score': metric}
#         return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

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

    # for epoch in range(epochs):
        # # train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # # update the learning rate
        # if lr_scheduler:
        #     lr_scheduler.step()
        # # evaluate on the test dataset
        # if val_dataloader:
        #     evaluate(model, val_dataloader, device=device)

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
        # evaluate on the test dataset
        if val_dataloader:
            evaluate(model, val_dataloader, device=device)

    #     if val_dataloader:
    #         pass
            # model.eval()
            # val_loss = 0.0
            # with torch.no_grad():
            #    boxes_true, boxes_pred = [], []
            #    lables_true, labels_pred = [], []
            #    for images, targets in tqdm(val_dataloader):
            #         if cuda_statement == True:
            #             images = list(image.to(device) for image in images)
            #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #         output = model(images)
            #         boxes_true.append(targets['boxes'].tolist())
            #         boxes_pred.append(output['boxes'].tolist())
            #         lables_true.append(targets['labels'].tolist())
            #         labels_pred.append(output['labels'].tolist())
            # mean_iou_list = iou_list(boxes_true, boxes_pred)
            # f1_per_class = f1_score(lables_true, labels_pred, average=None)
            # print(f'### {datetime.datetime.now()} ### [epoch {epoch + 1}]: train_time = {datetime.datetime.now()-start_time} | train_loss = {train_loss / len(train_dataloader)} | val_iou = {sum(mean_iou_list)/len(mean_iou_list)} | val_f1_per_class = {f1_per_class}')
        # else:
        #     print(f'### {datetime.datetime.now()} ### [epoch {epoch + 1}]: train_time = {datetime.datetime.now()-start_time} | train_loss = {train_loss / len(train_dataloader)}')

        # if lr_scheduler:
        #    lr_scheduler.step()

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
                image_id_list, predicted_box_list, true_box_list, true_label_list, predicted_label_list = [], [], [], [], []
                for images, targets in tqdm(test_dataloader):
                    images = list(image.to(cpu_device) for image in images)
                    targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
                    for target in list(targets):
                        image_id_list.append(target['image_id'].item())
                        true_label_list.append([int(t.detach().numpy()) for t in target['labels']])
                        true_box_list.append([t.detach().numpy().tolist() for t in target['boxes']])
                    if cuda_statement == True:
                        images = list(img.to(device) for img in images)
                        torch.cuda.synchronize()
                    outputs = model(images)
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    for output in outputs:
                        predicted_label_list.append([int(o.detach().numpy()) for o in output['labels']])
                        predicted_box_list.append([o.detach().numpy().tolist() for o in output['boxes']])
                output_df = pd.DataFrame()
                output_df['id'] = image_id_list
                output_df['true_label'] = true_label_list
                output_df['true_box'] = true_box_list
                output_df['predicted_label'] = predicted_label_list
                output_df['predicted_box'] = predicted_box_list
                output_df.to_csv(save_path+'model_output/model_output.csv')
        else:
            print(f'\nERROR: Cannot process a test set without specifying the save_path parameter')

    print(f'\n####### JOB FINISHED #######\n')

    return model
