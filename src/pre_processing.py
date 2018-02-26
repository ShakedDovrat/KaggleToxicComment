import os
import numpy as np
import pandas as pd


def create_all_text_file():
    train_file_csv = os.path.join('..', 'data', 'train.csv')
    test_file_csv = os.path.join('..', 'data', 'test.csv')
    output_file = os.path.join('..', 'data', 'all_data.txt')

    train = pd.read_csv(train_file_csv)
    test = pd.read_csv(test_file_csv)
    train = train[['comment_text']].copy()
    test = test[['comment_text']].copy()
    all_data = pd.concat([train, test])
    all_data = all_data.replace(r'\n', ' ', regex=True)
    np.savetxt(output_file, all_data.values, fmt='%s')


if __name__ == '__main__':
    create_all_text_file()
