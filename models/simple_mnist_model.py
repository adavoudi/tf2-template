from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from utils.utils import dummy_context_mgr
import tensorflow as tf


class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)

        with tf.distribute.MirroredStrategy().scope() \
                if self.config.trainer.multi_gpu else dummy_context_mgr():
            self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(28 * 28,)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )
