import logging
import time
import warnings

import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from functions_catalogue import (calculate_map, from_tsv_to_list,
                                 parse_model_outcome, prepare_data_for_ap,
                                 save_list_to_tsv_file)

# warnings
warnings.filterwarnings("ignore")


def evaluate_model(
    model: torchvision.models.detection.FasterRCNN,
    dataloader: torch.utils.data.DataLoader,
    main_dir: str,
    save_path: str,
    test: bool,
    num_classes: int,
    gpu: bool = False,
) -> None:
    evaluation_start_time = time.time()
    # switch to gpu if available
    cuda_statement = torch.cuda.is_available()
    logging.info(f"Cuda available: {cuda_statement}")
    device = (
        torch.device(torch.cuda.current_device())
        if cuda_statement and gpu
        else torch.device("cpu")
    )

    logging.info(f"Current device: {device}")
    # move model to the right device
    model.to(device)
    cpu_device = torch.device("cpu")
    # evaluation
    with torch.no_grad():
        (
            img_names_list,
            predicted_bboxes_list,
            predicted_labels_list,
            scores_list,
            new_sizes_list,
        ) = ([], [], [], [], [])
        if test:
            model_output, ground_truth = [], []
        for images, targets in tqdm(dataloader, desc="Evaluation"):
            if cuda_statement and gpu:
                images = [img.to(device) for img in images]
                torch.cuda.synchronize()
            else:
                images = [img.to(cpu_device) for img in images]
            targets = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in targets
            ]
            if test:
                ground_truth.append(targets)
            for target in list(targets):
                img_names_list.append(
                    int([t.detach().numpy() for t in target["image_name"]][0])
                )
                new_sizes_list.append(
                    [
                        t.detach().numpy().tolist()
                        for t in target["new_image_size"]
                    ]
                )
            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            if test:
                model_output.append(outputs)
            for output in outputs:
                predicted_labels_list.append(
                    [int(o.detach().numpy()) for o in output["labels"]]
                )
                predicted_bboxes_list.append(
                    [o.detach().numpy().tolist() for o in output["boxes"]]
                )
                scores_list.append(
                    [o.detach().numpy().tolist() for o in output["scores"]]
                )

    # save outputs
    logging.info("Saving model output")
    new_in_list = [f"{img}.jpg" for img in img_names_list]
    out_list = parse_model_outcome(
        new_in_list,
        f"{main_dir}/scraped_photos/",
        new_sizes_list,
        predicted_labels_list,
        scores_list,
        predicted_bboxes_list,
    )

    save_list_to_tsv_file(f"{main_dir}/data/{save_path}/out.tsv", out_list)
    save_list_to_tsv_file(f"{main_dir}/data/{save_path}/in.tsv", new_in_list)

    # mAP metric calculations
    if test:
        logging.info("Calculating mAP metric")

        out = from_tsv_to_list(f"{main_dir}/data/{save_path}/out.tsv")
        try:
            expected = from_tsv_to_list(
                f"{main_dir}/data/{save_path}/expected.tsv"
            )
        except:
            logging.exception(
                "File 'expected.tsv' not found, value of metric cannot be"
                " calculated."
            )
            raise FileNotFoundError()

        model_output, ground_truth = prepare_data_for_ap(out, expected)
        map_df = pd.DataFrame.from_dict(
            calculate_map(model_output, ground_truth, num_classes=num_classes),
            orient="index",
            columns=["AP"],
        )
        logging.info(f"Metric results:\n{map_df.to_string()}")

    logging.info(
        "Model evaluation completed, runtime:"
        f" {round(time.time()-evaluation_start_time,2)} sec."
    )
