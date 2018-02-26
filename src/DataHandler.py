import os
import pandas as pd
import numpy as np
import re
import language_check
import gensim
import word2vecReader


class DataHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        data_map = dict(raw=[], cleaned=[], vectored=[])
        self.data = dict(train=data_map.copy(), val=data_map.copy(), test=data_map.copy())
        self.tool = language_check.LanguageTool('en-US')

    def load(self):
        for data_type in ['train', 'test']:
            input_file = os.path.join(self.base_folder, data_type+'.csv')
            self.data[data_type]['raw'] = pd.read_csv(input_file)

    def clean(self, output_file_name=None):
        for data_type in ['train', 'test']:
            self.data[data_type]['cleaned'] = DataHandler._clean(self.data[data_type]['raw'])

        if output_file_name:
            all_comments = pd.concat([self.data['train']['cleaned'], self.data['test']['cleaned']])
            output_file_path = os.path.join(self.base_folder, output_file_name)
            np.savetxt(output_file_path, all_comments.values, fmt='%s')

    def train_word2vec(self, all_comments_file_path, pretrained_model_path):
        # # model = gensim.models.Word2Vec()
        # # model = gensim.models.Word2Vec.load(pretrained_model_path)
        # model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True, unicode_errors='ignore')
        # x=1
        # # model2 = word2vecReader.Word2Vec.load_word2vec_format(pretrained_model_path, binary=True)
        # # model2.train
        # gensim.models.Word2Vec
        # model.save_word2vec_format(os.path.join(os.path.dirname(pretrained_model_path), os.path.splitext(pretrained_model_path)[0] + '.txt'))
        # a = os.path.join(os.path.dirname(pretrained_model_path), os.path.splitext(pretrained_model_path)[0] + '2.txt')
        # model.save(a)
        # model2 = gensim.models.KeyedVectors.load(a)

        with open(all_comments_file_path, "r") as fid:
            comments = fid.readlines()
        # data = pd.read_csv(all_comments_file_path)
        # comments = data.values
        model = gensim.models.Word2Vec(size=400)
        model.build_vocab(comments)
        model.intersect_word2vec_format(pretrained_model_path, binary=True, unicode_errors='ignore')  # C binary format
        model.train(comments)
        x=1

    def analyze(self):
        self.data['train']['vectored'] = self.data['train']['cleaned'].apply(DataHandler.text_to_words)
        #self.data['train']['grammar_data'] =self.data['train']['cleaned'].apply(self.obtain_grammar_data)
        print(self.data['train']['vectored'])
        num_vectors = []
        words_list, _, _ = self.read_word2vec_output()
        for sen in self.data['train']['vectored']:
            num_vectors.append(self.convert_sentence_to_vector(sen, words_list))
        self.data['train']['input'] = num_vectors


    def obtain_grammar_data(self, raw_text):
        matches = self.tool.check(raw_text)
        num_of_spell_errors = [match.locqualityissuetype == 'misspelling' for match in matches]
        return np.array([len(maches), len(num_of_spell_errors)])

    def get_label_data_train(self):
        relevant_data = self.data['train']['raw']
        label_data = [relevant_data['toxic'], relevant_data['severe_toxic'], relevant_data['obscene'], relevant_data['threat'], relevant_data['insult'],
                      relevant_data['identity_hate']]

        return np.asarray(label_data)

    @staticmethod
    def text_to_words(raw_text, remove_stopwords=False):
        # 1. Remove non-letters, but including numbers
        letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
        # 2. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        # if remove_stopwords:
        #     stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
        #     meaningful_words = [w for w in words if not w in stops] # Remove stop words
        #     words = meaningful_words
        return words


    @staticmethod
    def _clean(data):
        data = data.copy()
        data = data.replace(r'\n', ' ', regex=True)
        data = data['comment_text'].apply(lambda x: x.lower())  # lower-case
        # ip_regex = r'^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$'
        ip_regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        data = data.replace(ip_regex, '_IP_', regex=True)
        data.fillna(value='_NONE_', inplace=True)  # fill nulls
        return data


    def read_word2vec_output(self, path=None):
        path = path or os.path.join(self.base_folder, 'all_data_word2vec_size_50_iter20.txt')
        f = open(path)
        lines = f.readlines()
        vocab_size = int(lines[0].split()[0])
        words_list = []
        vectors = []
        for line in lines[1:]:
            cur_vec = []
            parts = line.split()
            words_list.append(parts[0])
            for number in parts[1:]:
                cur_vec.append(float(number))
            cur_np_vec = np.array(cur_vec)
            vectors.append(cur_np_vec)
        return words_list, vectors, vocab_size

    @staticmethod
    def convert_sentence_to_vector(words, words_list):
        vec = map(lambda x: words_list.index(x), words)
        return vec

