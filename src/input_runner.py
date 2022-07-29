import os
import copy
import pandas as pd
import json
import logging
import click
from logs import Log
from functions_catalogue import save_list_to_tsv_file, rescale_annotations, calcMD5Hash, remove_coco_elem_if_in_list, remove_if_missed_annotations, input_transformer, get_input_statistics
from constants import General_args


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--main_dir",
    type=str,
    default=General_args.MAIN_DIR,
    help="Path to the level where this repository is stored.",
    show_default=True,
)

def prepare_input(main_dir):
    # check provided path
    assert os.path.exists(main_dir) == True
    
    # initialize logger
    logger = Log('input_runner')
    logger.log_start()

    # scraped images paths
    scraped_images = [img for img in os.listdir(f'{main_dir}/scraped_photos') if img != '.DS_Store']

    # read train and test annotation data
    with open(f"{main_dir}/source_annotations/train_80_percent.json") as jsonFile:
        coco_metadata_train = json.load(jsonFile)
        jsonFile.close()

    with open(f"{main_dir}/source_annotations/val_20_percent.json") as jsonFile:
        coco_metadata_test = json.load(jsonFile)
        jsonFile.close()


    # find images without annoations and remove them
    train_img_names = [coco_metadata_train['images'][i]['file_name'] for i in range(len(coco_metadata_train['images']))]
    test_img_names = [coco_metadata_test['images'][i]['file_name'] for i in range(len(coco_metadata_test['images']))]

    [scraped_images.remove(x) for x in train_img_names+test_img_names]

    # rescale annotations based on higher than actual resolution of scraped images
    logging.info('Rescaling source annotations to match scraped high-resolution images')
    coco_metadata_train = rescale_annotations(coco_metadata_train, main_dir)
    coco_metadata_test = rescale_annotations(coco_metadata_test, main_dir)

    # train validation split by md5 hash library
    set_names = [
        coco_metadata_train['images'][i]['file_name'].split('.')[0] for i in range(len(coco_metadata_train['images']))
    ]
    train_val_hash_list = list(pd.Series(set_names).apply(lambda x: calcMD5Hash(x, 0.08)))

    train_names, val_names = [], []

    # assign annotations to new splitted train val sets
    for i in range(len(train_val_hash_list)):
        if train_val_hash_list[i] == True:
            val_names.append(set_names[i])
        elif train_val_hash_list[i] == False:
            train_names.append(set_names[i])

    coco_metadata_val = copy.deepcopy(coco_metadata_train)

    coco_metadata_val = remove_coco_elem_if_in_list(coco_metadata_val, 'images', train_names, 'id')
    coco_metadata_val = remove_coco_elem_if_in_list(coco_metadata_val, 'annotations', train_names, 'image_id')

    coco_metadata_train = remove_coco_elem_if_in_list(coco_metadata_train, 'images', val_names, 'id')
    coco_metadata_train = remove_coco_elem_if_in_list(coco_metadata_train, 'annotations', val_names, 'image_id')

    # remove images without annoations 
    coco_metadata_train = remove_if_missed_annotations(coco_metadata_train)
    coco_metadata_val = remove_if_missed_annotations(coco_metadata_val)
    coco_metadata_test = remove_if_missed_annotations(coco_metadata_test)
    
    # store preprocessed annotatiosn in final list
    train_in, train_expected = input_transformer(coco_metadata_train, f'{main_dir}/scraped_photos/')
    val_in, val_expected = input_transformer(coco_metadata_val, f'{main_dir}/scraped_photos/')
    test_in, test_expected = input_transformer(coco_metadata_test, f'{main_dir}/scraped_photos/')
    
    train_stats, train_sum = get_input_statistics(train_expected)
    logging.info(f'Number of images in train set = {len(train_in)}')
    logging.info(f'Number of annotations in train set = {train_sum}')
    logging.info(f'Labels statistics for train set:\n{train_stats}')
        
    val_stats, val_sum = get_input_statistics(val_expected)
    logging.info(f'Number of images in validation set = {len(val_in)}')
    logging.info(f'Number of annotations in validation set = {val_sum}')
    logging.info(f'Labels statistics for validation set:\n{val_stats}')
        
    test_stats, test_sum = get_input_statistics(test_expected)
    logging.info(f'Number of images in test set = {len(test_in)}')
    logging.info(f'Number of annotations in test set = {test_sum}')
    logging.info(f'Labels statistics for test set:\n{test_stats}')
    
    # save data
    train_path = f"{main_dir}/data/train/"
    if not os.path.exists(train_path):
        logging.info('Path "data/train" doesn\'t exist, creating one')
        os.makedirs(train_path)

    save_list_to_tsv_file(f'{main_dir}/data/train/expected.tsv', train_expected)
    save_list_to_tsv_file(f'{main_dir}/data/train/in.tsv', train_in)

    # val:
    dev_path = f"{main_dir}/data/dev-0/"
    if not os.path.exists(dev_path):
        logging.info('Path "data/dev-0" doesn\'t exist, creating one')
        os.makedirs(dev_path)

    save_list_to_tsv_file(f'{main_dir}/data/dev-0/expected.tsv', val_expected)
    save_list_to_tsv_file(f'{main_dir}/data/dev-0/in.tsv', val_in)

    # test:
    test_path = f"{main_dir}/data/test-A/"
    if not os.path.exists(test_path):
        logging.info('Path "data/test-A" doesn\'t exist, creating one\n')
        os.makedirs(test_path)

    save_list_to_tsv_file(f'{main_dir}/data/test-A/in.tsv', test_in)
    save_list_to_tsv_file(f'{main_dir}/data/test-A/expected.tsv', test_expected)

    # end logger
    logger.log_end()

    
if __name__ == "__main__":
    prepare_input()
    