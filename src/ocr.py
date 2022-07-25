import os
import pathlib

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    os.popen("brew list tesseract | grep 'bin'").read().strip()
)

img_str = pytesseract.image_to_string(
    Image.open(f"{str(pathlib.Path(__file__).parent.resolve())}/ocr_test.png")
)

print(img_str)
