from DataHandler import *
from ToxicClassifier import *
from PerformanceEvaluator import *
import time


def main():
    data_handler = DataHandler(os.path.join('..', 'data'))
    data_handler.load()
    data_handler.clean()
    data_handler.analyze()

    config = Config()
    classifier = ToxicClassifier(data_handler, config)
    classifier.build_net()

    classifier.train()
    predictions = classifier.predict_on_test()

    analyzer = PerformanceEvaluator(data_handler, predictions)
    analyzer.analyze()
    analyzer.output_results('results{}.csv'.format(time.strftime('_%Y_%m_%d_%H_%M_%S')))


if __name__ == '__main__':
    main()
