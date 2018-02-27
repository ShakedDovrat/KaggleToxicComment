from keras.layers import Input, Dense, Activation, LSTM, Embedding
from keras.models import Model
import random as random
import keras
from keras import optimizers
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score


class Config:
    def __init__(self, embedding_dim=50, embedding_matrix=None, lstm_state_size=100,
                 batch_size=2 ** 12, num_epochs=6):
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.lstm_state_size = lstm_state_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = 0.01
        self.all_data = False


class ToxicClassifier:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.C = config
        [_, self.embedding_matrix, self.vocab_size] = self.data_handler.read_word2vec_output()

    def build_net(self):
        self.model = self._build_simple_model()
        optimizer = optimizers.Adam(lr=self.C.lr)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

    def _build_simple_model(self):
        # sequence_input = Input(shape=(None, 1))
        sequence_input = Input(shape=(None,), dtype='int32')
        embedding_layer = Embedding(self.vocab_size,
                                    self.C.embedding_dim,
                                    weights=None if self.embedding_matrix is None else
                                    [np.array(self.embedding_matrix)],
                                    trainable=True)
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(self.C.lstm_state_size, return_sequences=False)(embedded_sequences)
        x = Activation('relu')(x)
        x = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=sequence_input, outputs=x)
        return model

    def train(self):
        length = len(self.data_handler.data['train']['input'])
        split_idx = int(length * 0.8)
        x_all = self.data_handler.data['train']['input']
        x_all = pad_sequences(x_all, maxlen=100)
        y_all = self.data_handler.get_label_data_train()
        x = np.array(x_all[:split_idx])
        x_val = np.array(x_all[split_idx:])
        y = np.array(y_all[:split_idx])
        y_val = np.array(y_all[split_idx:])
        callbacks = [roc_callback(training_data=[x, y], validation_data=[x_val, y_val])]
        if self.C.all_data:
            x = x_all
            y = y_all
        self.model.fit(
            x=x,
            y=y,
            batch_size=self.C.batch_size,
            epochs=self.C.num_epochs,
            validation_data=(x_val, y_val),
            # callbacks=callbacks
        )

    def predict_on_test(self):
        x = np.array(self.data_handler.data['test']['input'])
        x = pad_sequences(x, maxlen=100)
        return self.model.predict(x=x, batch_size=self.C.batch_size)


class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print(
            '\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))) + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
