# Master thesis: The task of image retrieval in historical publications

<img src="https://img.shields.io/badge/python%20version-3.8.5-2D77D5" /> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" /> <img src="https://img.shields.io/badge/code%20style-black-black" />

Digitized historical publications (magazines, books, documents) abound in iconography (engravings, illustrations, photographs, diagrams). In view of this, a natural need arises to search for images that fulfill a given information need. The aim of this work is to create a training and testing set for the task of searching images in historical publications. The work will also develop a prototype neural networks-based search engine. Images will be retrieved and recognized along with the surrounding text.

I am using dataset described here: https://news-navigator.labs.loc.gov

### Instruction:

0. Clone the repository
1. Install requirements<em>\*1</em> and also clone one additional repository from here: https://github.com/scardine/image_size (put in the same directory as my repository)
2. Run "python src/scraper_runner.py" to obtain full-resolution photos from the Newspaper Navigator project
3. Run data processing notebook ("notebooks/data_preprocessing.ipynb") to create model input data from source annotations files (origin: https://github.com/LibraryOfCongress/newspaper-navigator/tree/master/beyond_words_data)
4. Run "python src/main.py" in command line to start training, make prediction or both (feel free to try various parameters values, run "python src/main.py --help" te see parameters description)
5. Run notebook creating output file basing on model output dataframe ("notebooks/create_out_file.ipynb") to save the model output in exactly the same structure as in notebook "data_preprocessing.ipynb"
6. Run notebook calculating average precision metric ("notebooks/metric_cal.ipynb") to calculate AP for each class and mAP value
7. Run data visualization notebook ("notebooks/data_visualization.ipynb") to visualize model predictions

<em>*1 Pytorch related packages are not included in requirements.txt, use following command to install them: "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html" or if you are using Mac with M1 chip "conda install pytorch torchvision torchaudio -c pytorch-nightly", otherwise training on cpu will take eternity, so I do not recommend it.</em>
