from keras.layers import Input, Dense, Activation, LSTM, Embedding
from keras.models import Model
import random as random
from keras import optimizers
import numpy as np


class Config:
    def __init__(self, vocab_size=1000, embedding_dim=100, embedding_matrix=None, lstm_state_size=100,
                 batch_size=2 ** 5, num_epochs=10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.lstm_state_size = lstm_state_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = 0.01


class ToxicClassifier:
    def __init__(self, data_handler, config):
        pass
        self.data_handler = data_handler
        # self.C = config

    def build_net(self):
        pass
        # self.model = self._build_simple_model()

        # optimizer = optimizers.Adam(lr=self.C.lr)
        # self.model.compile(loss="binary_crossentropy", optimizer=optimizer)

    def _build_simple_model(self):
        pass
        # sequence_input = Input()
        # embedding_layer = Embedding(self.C.vocab_size,
        #                             self.C.embedding_dim,
        #                             weights=None if self.C.embedding_matrix is None else [self.C.embedding_matrix],
        #                             trainable=True)
        # embedded_sequences = embedding_layer(sequence_input)
        # x = LSTM(self.C.lstm_state_size)(embedded_sequences)
        # x = Activation('relu')(x)
        # x = Dense(6, activation='sigmoid')(x)
        # model = Model(inputs=input, outputs=x)
        # return model

    def train(self):
        pass
        # x = self.data_handler['train']['input']
        # y = self.data_handler['train']['labels']
        # self.model.fit(
        #     x=x,
        #     y=y,
        #     batch_size=self.C.batch_size,
        #     epochs=self.C.num_epochs,
        #     validation_data=(self.data_handler['test']['input'], self.data_handler['test']['labels']))

    def evaluate(self):
        results = None
        L = len(self.data_handler.data['test']['cleaned'])
        results = np.random.rand(L, 6)
        return results

        # x = self.data_handler['test']['input']
        # y = self.data_handler['test']['labels']
        # return self.model.evaluate(x=x, y=y, batch_size=self.C.batch_size)

