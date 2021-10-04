#!/usr/bin/env python

# see tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlstm_pennylane import QLSTM

from matplotlib import pyplot as plt

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
ix_to_tag = {i:k for k,i in tag_to_ix.items()}

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}

# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

print(f"Vocabulary: {word_to_ix}")
print(f"Entities: {ix_to_tag}")

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0, backend='default.qubit'):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if n_qubits > 0:
            print(f"Tagger will use Quantum LSTM running on backend {backend}")
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, backend=backend)
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_logits, dim=1)
        return tag_scores
    

if __name__ == '__main__':
    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    parser = argparse.ArgumentParser("QLSTM Example")
    parser.add_argument('-E', '--embedding_dim', default=8, type=int)
    parser.add_argument('-H', '--hidden_dim', default=6, type=int)
    parser.add_argument('-Q', '--n_qubits', default=0, type=int)
    parser.add_argument('-e', '--n_epochs', default=300, type=int)
    parser.add_argument('-B', '--backend', default='default.qubit')
    args = parser.parse_args()

    print(f"Embedding dim:    {args.embedding_dim}")
    print(f"LSTM output size: {args.hidden_dim}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"Training epochs:  {args.n_epochs}")

    model = LSTMTagger(args.embedding_dim, 
                        args.hidden_dim, 
                        vocab_size=len(word_to_ix), 
                        tagset_size=len(tag_to_ix), 
                        n_qubits=args.n_qubits,
                        backend=args.backend)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    history = {
        'loss': [],
        'acc': []
    }
    for epoch in range(args.n_epochs):
        losses = []
        preds = []
        targets = []
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            labels = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

            probs = torch.softmax(tag_scores, dim=-1)
            preds.append(probs.argmax(dim=-1))
            targets.append(labels)
        avg_loss = np.mean(losses)
        history['loss'].append(avg_loss)

        #print("preds", preds)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        corrects = (preds == targets)
        accuracy = corrects.sum().float() / float(targets.size(0) )
        history['acc'].append(accuracy)

        print(f"Epoch {epoch+1} / {args.n_epochs}: Loss = {avg_loss:.3f} Acc = {accuracy:.2f}")

    # See what the scores are after training
    with torch.no_grad():
        input_sentence = training_data[0][0]
        labels = training_data[0][1]
        inputs = prepare_sequence(input_sentence, word_to_ix)
        tag_scores = model(inputs)

        tag_ids = torch.argmax(tag_scores, dim=1).numpy()
        tag_labels = [ix_to_tag[k] for k in tag_ids]
        print(f"Sentence:  {input_sentence}")
        print(f"Labels:    {labels}")
        print(f"Predicted: {tag_labels}")
    
    lstm_choice = "classical" if args.n_qubits == 0 else "quantum"

    #plt.figure(figsize=(6, 4))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(history['loss'], label=f"{lstm_choice} LSTM Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(history['acc'], label=f"{lstm_choice} LSTM Accuracy", color='tab:red')

    plt.title("Part-of-Speech Tagger Training")
    plt.ylim(0., 1.5)
    plt.legend(loc="upper right")

    plt.savefig(f"training_{lstm_choice}.png")
    plt.show()

    