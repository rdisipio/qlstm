#!/usr/bin/env python

import requests
import time
import pickle

import pandas as pd
import numpy as np

if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/sballas8/PoetRNN/master/data/haikus.csv"
    r = requests.get(url, allow_redirects=True)
    all_text = r.text
    open("haikus.csv", "w").write(all_text)

    df = pd.read_csv("haikus.csv", names=['haiku'])

    print(df)
    f = open('haikus.pkl', 'wb')
    pickle.dump(df, f)
