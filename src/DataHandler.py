import os
import pandas as pd
import numpy as np
import re


class DataHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        data_map = dict(raw=[], cleaned=[], vectored=[])
        self.data = dict(train=data_map.copy(), val=data_map.copy(), test=data_map.copy())

    def load(self):
        train_file = os.path.join(self.base_folder, 'train.csv')
        self.data['train']['raw'] = pd.read_csv(train_file)

    def clean(self):
        self.data['train']['cleaned'] = self.data['train']['raw']['comment_text'].copy()
        self.data['train']['cleaned'].fillna(value='none', inplace=True)

    def analyze(self):
        self.data['train']['vectored'] = self.data['train']['cleaned'].apply(DataHandler.text_to_words)
        print(self.data['train']['vectored'])

    @staticmethod
    def text_to_words(raw_text, remove_stopwords=False):
        # 1. Remove non-letters, but including numbers
        letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
        # 2. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        # if remove_stopwords:
        #     stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
        #     meaningful_words = [w for w in words if not w in stops] # Remove stop words
        #     words = meaningful_words
        return words
