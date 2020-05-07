# Sentiment classifications

The final working models are all in *.py files. You can run them directly from shell with python, e.g. '$ python train_baseline.py'. Progress is updated in the log.txt file, so you can track the progress of training. If you want to compare different models, uncomment the 'random.seed()' line at the beginning of each script.

The jupyter notebooks track the progress of this project as I was developing code. You can run them, but they will not have the most recent changes that I made to some utility functions.

### Baseline model

As my first model, I decided to make a simple baseline model so that I can compare the rest of my models to it. I took the idea from a pytorch tutorial and implement a model basen on nn.EmbeddingBag. It treats a sentence as a bag of words, computes the embeddings, averages them and uses two linear layers as a classification head. I use pretrained GloVe embeddings for all my models. I also wrote a simple learning rate finder (in utils.py) that helps pick a good learning rate (idea from fastai library).

First, I trained the model with the embedding weights frozen. Once I was happy with the results, I unfroze the weights to fine tune it.

### GRU

My first GRU model had a single linear layer attached to the last hidden layer. That was okay, but was overfitting quickly, So I added dropout right after the embedding and between gru layers - that helped!

Inspired by a post https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
and the ULMFiT paper https://arxiv.org/pdf/1801.06146.pdf, I tried out the concat-pooling head - average pooling and max pooling over the outputs from the uppermost gru layers, concatenated together with the last hidden state. It worked better than the previous version! I had to watch out to only pool over the outputs of non-padding tokens.

I also tried a self attention head from https://github.com/prakashpandey9/Text-Classification-Pytorch and https://arxiv.org/pdf/1703.03130.pdf. It gave really good results! It would be interesting to visualise the attention weights.

**
I realised the BucketIterator, which was supposed to collect examples with similar length into one batch in order to minimize padding, wasn't working at all. So I borrowed code from http://nlp.seas.harvard.edu/2018/04/03/attention.html#iterators. It sped up training by over a factor of two!
**
### CNN

I used code from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/CNN.pyand adjusted it a bit: added a few dropout layers, prittified the code. It worked very well, but seemed to overfit easily. The three convolutional layers are stacked horizontally, i.e. each is applied to the embedded sequence. I thought I would give it a try and stack similar layers vertically, like in a deep CNN. It seems to me that this version overfits less.

### To do:
* Try transformer networks. Get a pretrained network, like BERT, and fine-tune it on our dataset.
* Do k-fold cross validation and compare different models.
* Try different embeddings.

