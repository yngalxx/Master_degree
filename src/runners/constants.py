import pathlib


class Model:
    # editable
    WEIGHT_DECAY = 0
    LR_SCHEDULER = True
    LR_STEP_SIZE = 7
    LR_GAMMA = 0.5
    TRAINABLE_BACKBONE_LAYER = 5
    PRETRAINED = True
    LEARNING_RATE = 35e-5
    BATCH_SIZE = 20
    NUM_EPOCHS = 35


class Data:
    # editable
    NUM_WORKERS = 2
    RESCALE = "1024/1024"
    SHUFFLE = False
    CHANNEL = 1
    BBOX_FORMAT = "x0y0x1y1"
    CLASS_NAMES = [
        "photograph",
        "illustration",
        "map",
        "cartoon",
        "headline",
        "advertisement",
    ]
    NUM_CLASSES = len(CLASS_NAMES) + 1  # 1 for background
    COLOR_NAMES = [
        "lime",
        "orangered",
        "yellow",
        "deepskyblue",
        "cyan",
        "deeppink",
    ]
    # uneditable
    CLASS_CODING_DICT = {
        class_name: i + 1 for (i, class_name) in enumerate(CLASS_NAMES)
    }
    CLASS_COLORS_DICT = {
        class_name: color_name
        for (class_name, color_name) in zip(CLASS_NAMES, COLOR_NAMES)
    }


class General:
    # editable
    TRAIN_SET = True
    TEST_SET = True
    VAL_SET = True
    GPU = True
    # uneditable
    MAIN_DIR = "/".join(
        str(pathlib.Path(__file__).parent.resolve()).split("/")[:-2]
    )


class Output:
    # editable
    FORCE_SAVE_MODEL = False
    MIN_CONF_LEVEL = 0
    EXAMPLES = 5
    VAL_MAP_THRESHOLD = 0.8
    TEST_SET_EXPECTED = True
    UPDATE_EXISTING_OCR = False
    # uneditable
    MODEL_CONFIG_PATH = f"{General.MAIN_DIR}/model_config"
