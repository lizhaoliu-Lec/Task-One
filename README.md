# Task-One: Four micro torch projects.
- Powerful trainer can be implemented by inheriting BaseTrainer in trainers.py
- Clear module structures, trainers.py, models.py, datasets.py, utils.py...  

## Project 1: Image Classification via CNN
### support features
- clear console log
- tensorboard scalar and image visualization
### usage
```
# run model
python exp1_CNNImageCLassifier.py

# run tensorboard
cd runs/
tensorboard --logdir=./ --port=your_port (default 6006)
```


## Project 2: Image Detection and Segmentation via CNN
### support features
- clear console log
### usage
```
# run model
python exp2_CNNImageDetectorAndSegmentor.py
```


## Project 3: AG News Classification via Transformer
### Note 
due to the AG News implemented by torchtext will be downloaded from google drive, 
we need to explicitly download it and place it in root dir
### support features
- clear console log
- tensorboard scalar and text visualization
### usage
```
# run model
python exp3_TransformerTextClassifier.py

# run tensorboard
cd runs/
tensorboard --logdir=./ --port=your_port (default 6006)
```


## Project 4: Image Classification via [Stand-Alone Self-Attention in Vision Models](https://papers.nips.cc/paper/8302-stand-alone-self-attention-in-vision-models.pdf)
- clear console log
- tensorboard scalar and image visualization
### usage
```
# run model
python exp4_SelfAttImageClassifier.py

# run tensorboard
cd runs/
tensorboard --logdir=./ --port=your_port (default 6006)
```