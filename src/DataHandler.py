import os
import pandas as pd
import numpy as np
import re
import gensim


class DataHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        data_map = dict(raw=[], cleaned=[], vectored=[])
        self.data = dict(train=data_map.copy(), val=data_map.copy(), test=data_map.copy())

    def load(self):
        for data_type in {'train', 'test'}:
            input_file = os.path.join(self.base_folder, data_type+'.csv')
            self.data[data_type]['raw'] = pd.read_csv(input_file)

    def clean(self, output_file_name=None):
        for data_type in {'train', 'test'}:
            self.data[data_type]['cleaned'] = DataHandler._clean(self.data[data_type]['raw'])
        if output_file_name:
            all_comments = pd.concat([self.data['train']['cleaned'], self.data['test']['cleaned']])
            output_file_path = os.path.join(self.base_folder, output_file_name)
            np.savetxt(output_file_path, all_comments.values, fmt='%s')

    # def train_word2vec(self, pretrained_model_path='/home/bar/extDisk/kaggle/word2vec_twitter_model.bin'):
    #     gensim.models.Word2Vec

    def analyze(self):
        self.data['train']['vectored'] = self.data['train']['cleaned'].apply(DataHandler.text_to_words)

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

    @staticmethod
    def _clean(data):
        data = data.copy()
        data = data.replace(r'\n', ' ', regex=True)
        data = data['comment_text'].apply(lambda x: x.lower())  # lower-case
        data.fillna(value='NONE', inplace=True)  # fill nulls
        return data
