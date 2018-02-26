import os
from DataHandler import DataHandler


def main():
    data_handler = DataHandler(os.path.join('..', 'data'))
    data_handler.load()
    data_handler.clean('all_comments3.txt')
    # data_handler.train_word2vec(os.path.join('..', 'data', 'all_comments.txt'), '/home/bar/extDisk/kaggle/word2vec_twitter_model/word2vec_twitter_model.bin')


if __name__ == '__main__':
    main()
