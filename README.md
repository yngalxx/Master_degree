# Master thesis: The task of image retrieval in historical publications
Digitized historical publications (magazines, books, documents) abound in iconography (engravings, illustrations, photographs, diagrams). In view of this, a natural need arises to search for images that fulfill a given information need. The aim of this work is to create a training and testing set for the task of searching images in historical publications. The work will develop a prototype neural networks-based search engine. Images will be retrieved and recognized along with the surrounding text.

I am using dataset described here: https://news-navigator.labs.loc.gov

### Instruction:

1. Install requirements <b>*1</b> 
2. Run notebooks/data_preprocessing.ipynb to create model input directory 
3. Run src/python.py to start training/testing (feel free to try various parameters values, all of them are stored in dictionary right behind imports)
4. Run notebooks/create_out_file.ipynb to save the model output in exactly the same structure as in notebook data_preprocessing
5. Run notebooks/data_visualization.ipynb to visualize model predictions 

<b>*1</b> Torch related packages might fail during installation from requirements, so if sth like this happen try: "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html".

<b>Important!</b> Remember to change paths in each file :) 