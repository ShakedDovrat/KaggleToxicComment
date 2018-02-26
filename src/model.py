import keras

from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, Flatten, concatenate, MaxPool1D, LSTM, \
    Reshape, Bidirectional, TimeDistributed, Dropout, Concatenate, Embedding
from keras.models import Model

keras.layers.Embedding


class ToxicClassifier:
    def __init__(self, vocab_size, embedding_dim=100, embedding_matrix=None, lstm_state_size=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.lstm_state_size = lstm_state_size
        self.m = self.build_simple_model()

    def build_simple_model(self):
        sequence_input = Input()
        embedding_layer = Embedding(self.vocab_size,
                                    self.embedding_dim,
                                    weights=None if self.embedding_matrix is None else [self.embedding_matrix],
                                    trainable=True)
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(self.lstm_state_size)(embedded_sequences)
        x = Activation('relu')(x)
        x = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x)
        return model