import datetime
import warnings

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

# warnings
warnings.filterwarnings("ignore")


def model_predict(
    model: torchvision.models.detection.FasterRCNN,
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    gpu: bool = True,
) -> None:
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time} \n")
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    print(f"Cuda available: {torch.cuda.is_available()}")
    if cuda_statement == True:
        device = torch.device(torch.cuda.current_device())
    if gpu == False:
        device = "cpu"
    print(f"Current device: {device}\n")
    # move model to the right device
    model.to(device)
    print("###  Evaluaiting  ###")
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        (
            image_id_list,
            img_name_list,
            predicted_box_list,
            predicted_label_list,
            score_list,
            new_size_list,
        ) = ([], [], [], [], [], [])
        for images, targets in tqdm(dataloader):
            if cuda_statement == True:
                images = list(img.to(device) for img in images)
                torch.cuda.synchronize()
            else:
                images = list(img.to(cpu_device) for img in images)
            targets = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in targets
            ]
            for target in list(targets):
                image_id_list.append(target["image_id"].item())
                img_name_list.append(
                    int([t.detach().numpy() for t in target["image_name"]][0])
                )
                new_size_list.append(
                    [
                        t.detach().numpy().tolist()
                        for t in target["new_image_size"]
                    ]
                )
            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            for output in outputs:
                predicted_label_list.append(
                    [int(o.detach().numpy()) for o in output["labels"]]
                )
                predicted_box_list.append(
                    [o.detach().numpy().tolist() for o in output["boxes"]]
                )
                score_list.append(
                    [o.detach().numpy().tolist() for o in output["scores"]]
                )
        output_df = pd.DataFrame()
        output_df["image_id"] = image_id_list
        output_df["image_name"] = img_name_list
        output_df["image_name"] = output_df["image_name"].astype("int")
        output_df["predicted_labels"] = predicted_label_list
        output_df["predicted_boxes"] = predicted_box_list
        output_df["new_image_size"] = new_size_list
        output_df["score"] = score_list
        output_df = output_df.sort_values("image_name").reset_index(
            drop=True, inplace=False
        )
        output_df.to_csv(save_path)

    print(f"\n####### JOB FINISHED #######\n\n")
