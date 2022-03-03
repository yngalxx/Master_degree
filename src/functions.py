import csv
import numpy as np
import torch
import random


def dump_json(path, dict_to_save):
    jsonString = json.dumps(dict_to_save, indent=4)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def from_tsv_to_list(path, delimiter="\n"):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)
    expected = list(read_tsv)
    return [item for sublist in expected for item in sublist]


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_list_to_txt_file(path, output_list):
    with open(path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)


def save_list_to_tsv_file(path, output_list):
    with open(path, 'w', newline='') as f_output: 
        tsv_output = csv.writer(f_output, delimiter='\n')
        tsv_output.writerow(output_list)