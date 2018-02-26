import sklearn
import csv
import numpy as np

class PerformanceEvaluator:
    def __init__(self, data_handler, results):
        self.data_handler = data_handler
        self.results = results

    def analyze(self):
        # y_true = self.data_handler.data['test']['label']
        # y_score = self.results
        # score_total = sklearn.metrics.roc_auc_score(y_true, y_score)
        # print(score_total)
        pass

    def output_results(self, output_file):
        L = len(self.data_handler.data['test']['cleaned'])

        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow('id','toxic','severe_toxic','obscene','threat','insult','identity_hate')
            for line in range(L):
                line_id = self.data_handler.data['test']['raw']['id'][line]
                res = self.results[line, :]
                # ugly code, I'm sure there is a better way
                a, b, c, d, e, f = (np.array_str(res)[2:-1]).split('  ')
                writer.writerow([line_id, a, b, c, d, e, f])



