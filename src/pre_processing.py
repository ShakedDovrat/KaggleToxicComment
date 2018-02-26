import os
from DataHandler import DataHandler


def f():
    pretrained_model_path = '/home/bar/extDisk/kaggle/word2vec_twitter_model'


def main():
    data_handler = DataHandler(os.path.join('..', 'data'))
    data_handler.load()
    data_handler.clean('all_comments.txt')


if __name__ == '__main__':
    main()
