import os
import wget
import zipfile
import re
import logging
import argparse
import random
import time
import copy
import numpy as np
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data():
    # download data
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.listdir('data'):
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip', out='data/data.zip')
        with zipfile.ZipFile('data/data.zip', 'r') as myzip:
            myzip.extractall('data/')

    # separate lines into reviews and class labels
    path = os.path.abspath('data/sentiment labelled sentences')
    regex = re.compile(r'^(.*?)\s+(\d)$')
    reviews = []
    for file in ['imdb_labelled.txt', 'amazon_cells_labelled.txt', 'yelp_labelled.txt']:
        with open(os.path.join(path, file)) as f:
            for line in f:
                result = regex.match(line)
                reviews.append([result.group(1), int(result.group(2))])

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # remove sentences longer than 64 tokes
    for sentence, label in reviews.copy():
        encoded = tokenizer.encode(sentence, add_special_tokens=True)
        if len(encoded) > 64:
            reviews.remove([sentence, label])

    # tokenise and numericalise
    input_ids = []
    attention_masks = []
    labels = []

    for sent, label in reviews:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            max_length=64,
                            pad_to_max_length=True,
                            return_attention_mask=True,   # Construct attn. masks.
                            return_tensors='pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # make datasets and dataloaders
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset, tokenizer


def get_fold_dataloaders(ds, bs, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True)
    input_ids, attention_masks, labels = ds.tensors
    for train_index, val_index in kf.split(input_ids):
        train_dataset = TensorDataset(
            input_ids[train_index],
            attention_masks[train_index],
            labels[train_index]
        )
        val_dataset = TensorDataset(
            input_ids[val_index],
            attention_masks[val_index],
            labels[val_index]
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=bs)

        validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=bs)
        yield train_dataloader, validation_dataloader


def training(model, train_dl, val_dl):
    optimizer = AdamW(model.parameters(), lr=3e-5)
    epochs = 2
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    model.train()
    total_loss = 0
    start_time = time.time()
    for epoch in range(1, epochs+1):
        for i, batch in enumerate(train_dl, 1):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # validate 3 times per epoch
            if not i % (len(train_dl)//3):
                avg_train_loss = total_loss/(len(train_dl)//3)
                total_loss = 0
                avg_val_loss, avg_val_acc = validate(model, val_dl)
                logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, val_loss = {:.4f}, '
                             'val_accuracy : {:.3f}, time = {:.0f}s'
                             .format(epoch, i, avg_train_loss, avg_val_loss,
                                     avg_val_acc, time.time() - start_time))


def validate(model, val_dl):
    predictions = []
    gt = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            total_loss += loss.item()
            predictions.extend(logits.argmax(dim=1).tolist())
            gt.extend(b_labels.tolist())

    avg_loss = total_loss/len(val_dl)
    avg_acc = np.mean(np.array(predictions) == np.array(gt))
    return avg_loss, avg_acc


def cross_validate(model, ds, bs, n_folds):
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    iterator = get_fold_dataloaders(ds, bs, n_folds)
    original_model = model
    for i, (train_dl, val_dl) in enumerate(iterator, 1):
        logging.info('Cross-validating, fold : {}/{}'.format(i, n_folds))
        model = copy.deepcopy(original_model)
        training(model, train_dl, val_dl)
        for dl, dl_id in zip([train_dl, val_dl], ['train', 'val']):
            loss, acc = validate(model, dl)
            losses[dl_id].append(loss)
            accuracies[dl_id].append(acc)
    logging.info('--- Cross-validation statistics ---')
    logging.info('Training loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['train'])/n_folds, sum(accuracies['train'])/n_folds))
    logging.info('Validation loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['val'])/n_folds, sum(accuracies['val'])/n_folds))


def main():
    # Setup logger
    logging.basicConfig(filename='log.txt',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s : %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info('Running file : {}'.format(__file__))

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--randomseed',
                        type=int,
                        default=None,
                        help='random seed')
    parser.add_argument('-cv', '--crossvalidate',
                        action='store_true',
                        help='cross-validation does not save the model at the end!')
    parser.add_argument('-k',
                        type=int,
                        default=10,
                        help='number of folds, default=10')
    args = parser.parse_args()

    # get dataset
    dataset, tokenizer = get_data()

    if args.randomseed is not None:
        random.seed(args.randomseed)
        np.random.seed(args.randomseed)
        torch.manual_seed(args.randomseed)
        # torch.cuda.manual_seed_all(args.randomseed)

    # create model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
        ).to(device)
    bs = 16

    # cross validation routine
    if args.crossvalidate:
        cross_validate(model, dataset, bs, args.k)
    # single split routine
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dl = DataLoader(
                    train_dataset,
                    sampler=RandomSampler(train_dataset),
                    batch_size=bs
                )

        val_dl = DataLoader(
                    val_dataset,
                    sampler=SequentialSampler(val_dataset),
                    batch_size=bs
                )

        training(model, train_dl, val_dl)
        logging.info('--- Final statistics ---')
        logging.info('Training loss : {:.4f}, accuracy {:.4f}'
                     .format(*validate(model, train_dl)))
        logging.info('Validation loss : {:.4f}, accuracy {:.4f}'
                     .format(*validate(model, val_dl)))
        if not os.path.exists('models'):
            os.makedirs('models')
        logging.info('Model saved to: models/model_BERT.pt')
        torch.save(model.state_dict(), 'models/model_BERT.pt')


if __name__ == '__main__':
    main()
