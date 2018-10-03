# ROCCO - Framework to train neural networks for Semantic Segmentation

## Motivation

This is a modular framework to ease training of neural networks for semantic segmentation. 
A separation of network design, training and evaluation is reflected by the software architecture. 
All parameters to describe the training procedure and evaluation phase are stored in a YAML config file and serve for documentation and reproduction of results.
Therefore, changing the network architecture will ideally affect only a single line in the config file. 

New networks can easily be integrated into the framework as well as custom datasets, augmentation strategies etc. See the section **Design Principles** for further details. 

## Design Principles

When designing a neural network for semantic segmentation, you will likely ask yourself the following questions
 - How should my architecture look like?
 - Which state of the art architectures are available?
 - Which data will I use to train my network?
 - How should I preprocess my images to feed them into the network?
 
These questions are further followed by
 - Which data augmentation can I apply to my dataset?
 
and in the end after training
 - How can I evaluate my network on (multiple) dataset(s)
 
This frameworks aims to provide an easy to use solution for all those questions and automates evaluation.

### Dependencies

Although the currently available networks are implemented in Keras, the framework is designed such that it is framework-agnostic.
For data augmentation, [img_aug]() is used.

## Installation

    pip install -r requirements.txt 

## Setup

### Dataset

#### Cityscapes

     export CITYSCAPES_DATASET="..."
     ./thirdparty/cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
     
## Usage

There is a single entrypoint for all tasks. See the output of

    python main.py -h
    
for a complete list of options.

To train your network, you need to provide a config file like the one you find in the toplevel directory of this repo.
Then, simply execute

    python main.py train --config='...'

That's it.

To test and evaluate your network, simply replace *train* by those commands as the first argument.

## Tests



# ToDo

 - [ ] Add tests for all components
