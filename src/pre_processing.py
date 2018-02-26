import os
import numpy as np
from DataHandler import DataHandler
from PerformanceEvaluator import PerformanceEvaluator


def main():
    data_handler = DataHandler(os.path.join('..', 'data'))
    data_handler.load()
    # data_handler.clean('all_comments3.txt')
    # # data_handler.train_word2vec(os.path.join('..', 'data', 'all_comments.txt'), '/home/bar/extDisk/kaggle/word2vec_twitter_model/word2vec_twitter_model.bin')

    num_test_data = data_handler.data['test']['raw'].shape[0]
    num_classes = 6
    dummy_results = np.random.rand(num_test_data, num_classes)
    pe = PerformanceEvaluator(data_handler, dummy_results)
    pe.output_results('../data/dummy_results.csv')


if __name__ == '__main__':
    main()
