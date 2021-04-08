#!/usr/bin/env python

import requests
import time
import pickle

import pandas as pd
import numpy as np

vocab = [
    'pad', 'sos', 'eos',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'x', 'z',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    ' ', '!', '?', '&', '.', ':', ';', ',', '-', '\n', '\'', '\"', '/', '(', ')',
]

char2id = {c: i for i,c in enumerate(vocab)}
id2char = {i: c for i,c in enumerate(vocab)}


def tokenize(text: str):
    text = text.lower().strip()
    tokens_str = [x for x in text]
    token_ids = [char2id[c] for c in tokens_str]
    token_ids = [char2id['sos']] + token_ids + [char2id['eos']]
    return token_ids


if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/sballas8/PoetRNN/master/data/haikus.csv"
    r = requests.get(url, allow_redirects=True)
    all_text = r.text

    all_text = all_text.replace('&nbsp', " ")
    all_text = all_text.replace('[', " ")
    all_text = all_text.replace(']', " ")
    all_text = all_text.replace('{', " ")
    all_text = all_text.replace('}', " ")

    open("haikus.csv", "w").write(all_text)

    df = pd.read_csv("haikus.csv", names=['haiku'])

    df['token_ids'] = df['haiku'].apply(lambda x: tokenize(x))
    df['n_tokens'] = df['token_ids'].apply(lambda x: len(x))

    print(df)

    data = {
        'vocab': vocab,
        'df': df
    }

    f = open('haikus.pkl', 'wb')
    pickle.dump(data, f)
