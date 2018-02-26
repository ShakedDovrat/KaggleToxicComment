import os
import pandas as pd
import numpy as np

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
        self.data['train']['vectored'] = self.data['train']['cleaned']


