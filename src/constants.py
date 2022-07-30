import pathlib


class Model:
    WEIGHT_DECAY = 0
    LR_SCHEDULER = True
    LR_STEP_SIZE = 5
    LR_GAMMA = 0.4
    TRAINABLE_BACKBONE_LAYER = 5
    PRETRAINED = True
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 20


class Data:
    NUM_WORKERS = 2
    RESCALE = "1000/1000"
    SHUFFLE = False
    CHANNEL = 1
    BBOX_FORMAT = "x0y0x1y1"
    NUM_CLASSES = 7  # 1 for background
    CLASS_COLORS_DICT = {
        "photograph": "lime",
        "illustration": "orangered",
        "map": "yellow",
        "cartoon": "deepskyblue",
        "headline": "cyan",
        "advertisement": "deeppink",
    }
    CLASS_CODING = {
        "photograph": 1,
        "illustration": 2,
        "map": 3,
        "cartoon": 4,
        "headline": 5,
        "advertisement": 6,
    }


class General:
    MAIN_DIR = "/".join(
        str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1]
    )
    TRAIN_SET = True
    TEST_SET = True
    VAL_SET = True
    GPU = True


class Output:
    MODEL_CONFIG_PATH = f"{General.MAIN_DIR}/model_config"
    FORCE_SAVE_MODEL = False
    MIN_CONF_LEVEL = 0
    EXAMPLES = 5
