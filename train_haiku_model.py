#!/usr/bin/env python

import pickle

import numpy as np
import pandas as pd

#from nltk import ngrams
from sklearn.model_selection import train_test_split 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


from qlstm_pennylane import QLSTM


def get_ngrams(X, n):
    ngrams = [X[i:i+n] for i in range(len(X)-n+1)]
    return ngrams


class HaikuLM(pl.LightningModule):
    def __init__(self, 
                embed_dim: int,
                vocab_size: int,
                hidden_dim: int,
                n_qubits: int=0,
                lr=1e-3,
                backend: str='default.qubit'):
        super(HaikuLM, self).__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.backend = backend
        self.lr = lr

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        #self.lstm = QLSTM(embed_dim, hidden_dim)
        self.hidden2id = nn.Linear(hidden_dim, vocab_size)
        #self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        #self.train_acc = pl.metrics.Accuracy()
        #self.valid_acc = pl.metrics.Accuracy(compute_on_step=False)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        last_lstm_timestamp = lstm_out[:, -1, :]
        tag_logits = self.hidden2id(last_lstm_timestamp)
        #tag_logits = self.hidden2id(lstm_out.view(len(sentence), -1))
        #tag_scores = F.log_softmax(tag_logits, dim=1)
        return tag_logits
    
    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        preds = self.forward(input_ids)

        #print(">>>", torch.argmax(preds, dim=-1))
        #print(">>>", target_ids)
        #print("---")
        loss = self.criterion(preds, target_ids)
        self.log('loss', loss)
        #loss, acc = self._training(batch)
        #metrics = {'loss': loss, 'acc': acc}
        #self.log_dict(metrics, loss, on_step=True, on_epoch=True)

        # The `preds` should be probabilities, but values were detected outside of [0,1] range.
        #self.train_acc(preds, target_ids)
        #self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    '''
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
    '''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 32
    WINDOW_SIZE = 10
    EMBED_DIM = 4
    HIDDEN_DIM = 8
    N_QUBITS = 4
    BACKEND = 'default.qubits'
    N_GPUS = 0

    f = open('haikus.pkl', 'rb')
    data = pickle.load(f)

    df = data['df']
    vocab = data['vocab']
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    all_entries = df['token_ids']
    dataset = []
    for row in all_entries:
        ngrams = get_ngrams(row, WINDOW_SIZE)
        dataset.extend(ngrams)
    dataset = np.array(dataset, dtype=np.int64)
    
    X = dataset[:, :WINDOW_SIZE-1]
    y = dataset[:, -1]

    print(X.shape, y.shape)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

    dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
    test_size = int(0.3 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    train_ds = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_ds = DataLoader(test_dataset, num_workers=2)

    model = HaikuLM(
        embed_dim=EMBED_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_qubits=N_QUBITS,
        backend=BACKEND)
    
    trainer = pl.Trainer(max_epochs=EPOCHS, gpus=N_GPUS)
    trainer.fit(model, train_ds, test_ds)

    data = [
        model.state_dict(),
        {
            'embed_dim': EMBED_DIM,
            'vocab_size': VOCAB_SIZE,
            'hidden_dim': HIDDEN_DIM,
            'n_qubits': N_QUBITS
        }
    ]
    torch.save(data, "haiku_model.pt")
