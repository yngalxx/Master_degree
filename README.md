# Master thesis: The task of image retrieval in historical publications
<img src="https://img.shields.io/badge/Python%20version-3.8.5-2D77D5" /> <img src="https://img.shields.io/badge/CUDA%20version-11.4-7ED52D" /> <img src="https://results.pre-commit.ci/badge/github/pre-commit/pre-commit/main.svg" /> <img src="https://camo.githubusercontent.com/d91ed7ac7abbd5a6102cbe988dd8e9ac21bde0a73d97be7603b891ad08ce3479/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64652532307374796c652d626c61636b2d3030303030302e737667" />
Digitized historical publications (magazines, books, documents) abound in iconography (engravings, illustrations, photographs, diagrams). In view of this, a natural need arises to search for images that fulfill a given information need. The aim of this work is to create a training and testing set for the task of searching images in historical publications. The work will also develop a prototype neural networks-based search engine. Images will be retrieved and recognized along with the surrounding text.

I am using dataset described here: https://news-navigator.labs.loc.gov

### Instruction:

0. Clone this repository
1. Install requirements <em>*1</em>
2. Run "python src/scraper_runner.py" to obtain full-resolution photos from the Newspaper Navigator project
3. Run data processing notebook ("notebooks/data_preprocessing.ipynb") to create model input data from source annotations files (origin: https://github.com/LibraryOfCongress/newspaper-navigator/tree/master/beyond_words_data)
4. Run "python src/main.py" in command line to start training, make prediction or both (feel free to try various parameters values, run "python src/main.py --help" te see parameters description)
5. Run notebook creating output file basing on model output dataframe ("notebooks/create_out_file.ipynb") to save the model output in exactly the same structure as in notebook "data_preprocessing.ipynb"
6. Run notebook calculating average precision metric ("notebooks/metric_cal.ipynb") to calculate AP for each class and mAP value
7. Run data visualization notebook ("notebooks/data_visualization.ipynb") to visualize model predictions

<em>*1 Torch related packages might fail during installation from requirements, so if sth like this happen try: "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html".</em>
