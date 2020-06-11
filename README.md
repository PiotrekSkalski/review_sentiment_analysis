# IMDb movie reviews - sentiment analysis

This project aims to make a few lightweight classifiers that train quikly and are fast at evaluation.

# How to run it?

Follow the jupyter notebooks to train interactively and visualise the learning 
curves and activations. Alternatively, run the ``*.py`` files, e.g.

    $ python GRU/train_GRU_selfattention.py --seed 35 --save-to GRU_model.pt

Make sure to install libraries from ``requirements.txt``

# Models

All of the below models use pretrained [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) embeddings of 840B tokens from Common Crawl. Training procedure follows the guidelines from [ULMFiT](https://arxiv.org/pdf/1801.06146.pdf) paper: train the model for a bit with the emveddings frozen, then fine-tune the whole model together with embedding weights. I also use slanted triangular cyclic learning rate scheduler.

I did not do any search over the hyperparameter space. Instead, I picked the learning rate with the help of a [``lr_finder``](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) tool, and tried a few different dropout rates and number of hidden layers until my learning curves looked acceptable, i.e. no signs of under/over-fitting.

## Bag of embeddings (baseline)

It is the simplest model over the embeddings - it just averages the word vectors for each token in a sentences and applies a linear layer on top.

* Test set accuracy: ``88.4% ``

## GRU with concat pooling

This model is based on the ["Revisiting LSTM Networks for Semi-Supervised
Text Classification via Mixed Objective Function"](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_46.pdf). Instead of LSTM, I use GRU. Also, on top I apply a concat pooling layer, which is a concatenation of max pooling and average pooling (over time). 

* Test set accuracy: ``90.8%``

## GRU with self attention

This is a fuse of RNN and attention based on ["A Structured Self-Attentive Sentence Embedding"](https://arxiv.org/pdf/1703.03130.pdf). Instead of LSTN, I use GRU. This model has an extra trick, which is penalisation trick added to the loss function that prevents the attention from learning redundant weights.

* Test set accuracy: ``91.6%``

## Shallow CNN

This model is based on one of the first papers on the use of convolutional layers for NLP ["Convolutional Neural Networks for Sentence Classification"](https://arxiv.org/pdf/1408.5882.pdf). It's faster than the RNNs but a bit less accurate.

* Test set accuracy: ``89.7%``

## Deep CNN

I applied a model described in ["Very Deep Convolutional Networks
for Text Classification"](https://arxiv.org/pdf/1606.01781.pdf). Due to its depth, it's the slowest one, however, doesn't perform any better than the self-attention model.

* Test set accuracy: ``91.5%``

# Visualisations

One can visualise which parts of the sentences contribute most to model's confidence with the following techniques:

* In the GRU model with self-attention, we can use the attention weights as indicators.
* In models with convolutions, I used the idea of Grad-CAM from["Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization"](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).

We can use these heatmaps to confirm that models are working correctly or, as the paper suggests, debug and debias them.

![heatmap](/imgs/gru_attn.jpg)
![heatmap](/imgs/DCNN_heatmap.jpg)
