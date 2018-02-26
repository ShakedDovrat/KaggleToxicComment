import sklearn
import pandas as pd


class PerformanceEvaluator:
    def __init__(self, data_handler, results):
        self.data_handler = data_handler
        self.results = results

    def analyze(self):
        y_true = self.data_handler.data['test']['label']
        y_score = self.results
        score_total = sklearn.metrics.roc_auc_score(y_true, y_score)
        print(score_total)

    def output_results(self, output_file):
        test_df = self.data_handler.data['test']['raw']
        x = test_df['id']
        x2 = pd.DataFrame(self.results, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
        x3 = pd.concat([x, x2], axis=1)
        x3.to_csv(output_file, index=False)
