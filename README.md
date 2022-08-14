# Master thesis: The task of image retrieval in historical publications

<p>
    <img src="https://img.shields.io/badge/python%20version-3.8.5-2D77D5" /> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" /> <img src="https://img.shields.io/badge/github%20actions-enabled-success" /> <img src="https://img.shields.io/badge/code%20style-black-black" /> <img src="https://img.shields.io/badge/import%20format-pycln%20%7C%20isort-orange" /> <a href="https://codeclimate.com/github/yngalxx/Master_degree/maintainability"> <img src="https://api.codeclimate.com/v1/badges/32d0d5c35ad32b251d6c/maintainability" /></a> <a href="https://lgtm.com/projects/g/yngalxx/Master_degree/context:python"><img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/yngalxx/Master_degree.svg?logo=lgtm&logoWidth=18"/></a> <a href="https://lgtm.com/projects/g/yngalxx/Master_degree/alerts/"><img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/yngalxx/Master_degree.svg?logo=lgtm&logoWidth=18"/></a>
</p>

Digitized historical publications (magazines, books, documents) abound in iconography (engravings, illustrations, photographs, diagrams). In view of this, a natural need arises to search for images that fulfill a given information need. The aim of this work is to create a training and testing set for the task of searching images in historical publications. The work will also develop a prototype neural networks-based search engine. Images will be retrieved and recognized along with the surrounding text.

I am using dataset described here: https://news-navigator.labs.loc.gov

### Instruction:

0. Before use:   
    - clone the repository,
    - install requirements,
    - install Tesseract OCR using homebrew (run "brew install tesseract"),
    - install spaCy language core for english (run "python -m spacy download en_core_web_sm"),
    - run "python setup.py install"
1. Run "python scraper_runner.py" to obtain high-resolution images from the Newspaper Navigator project.
2. Run "python preprocessing_runner.py" to create model input data from source annotations files.
3. Run "python model_runner.py" to start training, evaluation or both (feel free to try various argument values).
4. Run "python metric_runner.py" to calculate the average precision (AP) for each class, as well as its mean value (mAP).
5. Run "python visualization_runner.py" to visualize several random model predictions.
6. Run "python predict_runner.py" to make prediction on your own single newspaper image.
7. Run "python ocr_runner.py" to crop visual content from test set of original images using resulting bbox predictions, apply OCR on them and store results in SQLite database.
8. Run "python gui_runner.py" to launch GUI for full-text search through OCR results on predicted visual content.

**IMPORTANT:**
- If you intend to use GPU install Pytorch using following command: "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html".
- Each script in directory named 'runner' is a command line application (despite of 'constants.py', where you can edit default arguments). Run each with argument '--help' to see the description of the other arguments.
- Valid paths are generated automatically, but you can provide specific ones using click arguments in the command line for each runner.


### Gallery: 
![Alt text](gui_graphics/Results.png?raw=true "Application GUI")