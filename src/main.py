from DataHandler import *
from ToxicClassifier import *
from model import ToxicClassifier
from PerformanceEvaluator import *


def main():
    data_handler = DataHandler('..\data')
    data_handler.load()
    data_handler.clean()
    data_handler.analyze()

    classifier = ToxicClassifier(data_handler)
    classifier.build_net()
    classifier.train()
    classifier.evaluate()

    analyzer = PerformanceEvaluator(data_handler)
    analyzer.analyze()
    analyzer.output_results()


if __name__ == '__main__':
    main()
