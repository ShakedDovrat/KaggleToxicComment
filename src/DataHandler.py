import os
import pandas as pd
import numpy as np
import re
import language_check



class DataHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        data_map = dict(raw=[], cleaned=[], vectored=[])
        self.data = dict(train=data_map.copy(), val=data_map.copy(), test=data_map.copy())
        self.tool = language_check.LanguageTool('en-US')

    def load(self):
        train_file = os.path.join(self.base_folder, 'train.csv')
        test_file = os.path.join(self.base_folder, 'test.csv')
        self.data['train']['raw'] = pd.read_csv(train_file)
        self.data['test']['raw'] = pd.read_csv(test_file)

    def clean(self, output_file_name=None):
        self.data['train']['cleaned'] = DataHandler._clean(self.data['train']['raw'])
        self.data['test']['cleaned'] = DataHandler._clean(self.data['test']['raw'])
        if output_file_name:
            all_comments = pd.concat([self.data['train']['cleaned'], self.data['test']['cleaned']])
            output_file_path = os.path.join(self.base_folder, output_file_name)
            np.savetxt(output_file_path, all_comments.values, fmt='%s')

    # def train_word2vec(self, pretrained_model_path='/home/bar/extDisk/kaggle/word2vec_twitter_model.bin'):
    #     gensim.models.Word2Vec

    def analyze(self):
        self.data['train']['vectored'] = self.data['train']['cleaned'].apply(DataHandler.text_to_words)
        #self.data['train']['grammar_data'] =self.data['train']['cleaned'].apply(self.obtain_grammar_data)
        self.data.to_csv(os.path.join(self.base_folder, 'train_analysis.csv'))
        print(self.data['train']['vectored'])

    def obtain_grammar_data(self, raw_text):
        matches = self.tool.check(raw_text)
        num_of_spell_errors = [match.locqualityissuetype == 'misspelling' for match in matches]
        return np.array([len(matches), len(num_of_spell_errors)])

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

