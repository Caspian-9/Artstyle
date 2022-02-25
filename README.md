# ArtstyleClassifier
AI model made with Tensorflow Keras that can recognize 3 different art styles: cubism, Ghibli, and impressionism

artstyle.py is the Python file for the whole program. artstyle.ipynb is the Jupyter Notebook file for artstyle.py

## Instructions (for Windows):

Requirements
- Python 3
- Jupyter Notebook
- matplotlib
- Anaconda


## Commands: 

Train model with \[num] of epochs:
```
python artstyle.py train --dataset="content\datasets\photos" --epochs=[num]
```

Evaluate model: 
```
python artstyle.py evaluate --model="model/[filename of model].h5" --testdir="content\datasets\photos\test"
```

Use the model to classify a set of images: 
```
python artstyle.py classify --model="model/[filename of model].h5" --image="[directory of images to classify]"
```
