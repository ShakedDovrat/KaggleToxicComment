from DataHandler import *
from ToxicClassifier import *
from model import ToxicClassifier


def main():
    data_handler = DataHandler('..\data')
    data_handler.Load()
    data_handler.Clean()
    data_handler.Analyze()

    classifier = ToxicClassifier(data_handler)
    classifier.BuildNet()
    classifier.Train()
    classifier.Evaluate()
    classifier.AnalyzePerformance()


if __name__ == '__main__':
    main()
