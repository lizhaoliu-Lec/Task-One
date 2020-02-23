# Task-One: Four micro torch projects.
- Powerful trainer can be implemented by inheriting BaseTrainer in trainers.py
- Clear module structures, trainers.py, models.py, datasets.py, utils.py...  

## Project 1: Image Classifier via CNN
### support features
- clear console log
- tensorboard scalar and image visualization
### usage
```
# run model
python image_classifier_exp1.py

# run tensorboard
cd runs/
tensorboard --logdir=./ --port=your_port (default 6006)
```


## Project 2: Image Detector and Segmentor via CNN
### support features
- clear console log
### usage
```
# run model
python detectorAndsegmentor_exp2.py
```


## Project 3: AG News Classification via Transformer
### Note 
due to the AG News implemented by torchtext will be downloaded from google drive, 
we need to explicitly download it and place it in your root dir
### support features
- clear console log
- tensorboard scalar and text visualization
### usage
```
# run model
python text_classifier_exp3.py

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
python selfAtt_image_classifier_exp4.py

# run tensorboard
cd runs/
tensorboard --logdir=./ --port=your_port (default 6006)
```