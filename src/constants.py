import pathlib


class Model_args:
    WEIGHT_DECAY = 0
    LR_SCHEDULER = True
    LR_STEP_SIZE = 5
    LR_GAMMA = 0.4
    TRAINABLE_BACKBONE_LAYER = 5
    PRETRAINED = True
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 20


class Data_args:
    NUM_WORKERS = 2
    RESCALE = "1000/1000"
    SHUFFLE = False
    NUM_CLASSES = 7
    CHANNEL = 1
    BBOX_FORMAT = "x0y0x1y1"


class General_args:
    MAIN_DIR = "/".join(
        str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1]
    )
    TRAIN_SET = True
    TEST_SET = True
    VAL_SET = True
    GPU = True


class Output_args:
    MODEL_CONFIG_PATH = (
        f"{General_args.MAIN_DIR}/model_config/model_config.json"
    )
    FORCE_SAVE_MODEL = True
    MIN_CONF_LEVEL = 0
