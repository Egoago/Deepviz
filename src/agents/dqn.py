import logging
import time

import keras.models
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import layers
from tqdm import tqdm

from src.agents import Agent


class DQN(Agent):
    def __init__(self, *kwargs):
        super(DQN, self).__init__(*kwargs)
        self.gamma = 0.95
        self.model = self.__build_model__()
        self.target_model = self.__build_model__()
        self.loss = Huber()
        self.optimizer = Adam(learning_rate=0.00015, clipnorm=1.0)

    def __build_model__(self):
        inputs = layers.Input(shape=self.state_shape)

        layer = layers.Conv2D(8, 5, strides=2, activation="relu")(inputs)
        layer = layers.Conv2D(16, 3, strides=2, activation="relu")(layer)
        layer = layers.BatchNormalization()(layer)

        path1 = layers.Conv2D(8, 5, strides=1, activation="relu")(layer)
        path1 = layers.MaxPooling2D()(path1)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.Flatten()(path1)

        path2 = layers.Conv2D(16, 4, strides=2, activation="relu")(layer)
        path2 = layers.Conv2D(16, 3, strides=1, activation="relu")(path2)
        path2 = layers.Conv2D(26, 3, strides=1, activation="relu")(path2)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.Flatten()(path2)

        residual = layers.Flatten()(layer)
        layer = layers.add([path1, path2])
        layer = layers.concatenate([layer, residual])

        layer = layers.Dense(256, activation="relu")(layer)
        layer = layers.Dense(128, activation="relu")(layer)
        outputs = layers.Dense(self.action_num, activation="linear")(layer)
        model = keras.Model(inputs=inputs, outputs=outputs)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print(model.summary())
        return model

    def move(self, state):
        action = self.explore()
        assert state.shape == self.state_shape
        if action is not None:
            return action
        return self.exploit(state)

    def exploit(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        actions = self.model(state, training=False)[0]
        return tf.argmax(actions).numpy()

    def __replay__(self):
        replay_size = 32
        if replay_size < self.replay_memory.length:
            states, actions, rewards, next_states, dones = self.replay_memory.sample(32)
            next_rewards = self.target_model.predict(next_states)
            target_q = rewards + self.gamma * np.max(next_rewards, axis=1)
            target_q = (1 - dones) * target_q + dones * rewards
            self.__train_model__(actions, states, target_q)

    def __train_model__(self, actions, states, target_q):
        masks = tf.one_hot(actions, self.action_num)
        with tf.GradientTape() as tape:
            current_q = self.model(states)
            masked_q = tf.reduce_sum(tf.multiply(current_q, masks), axis=1)
            loss = self.loss(target_q, masked_q)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, env):
        state = env.reset()
        self.episodes = []
        episode_steps = 0
        for total_steps in tqdm(range(self.max_processed_steps), desc='Training', unit='step'):
            episode_steps += 1
            action = self.move(state)
            next_state, reward, done = env.step(env.actions[action])

            if episode_steps > 30:
                self.replay_memory.register((state, action, reward, next_state, done))
                if total_steps % 4 == 0:
                    self.__replay__()
                if total_steps % 10000 == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if total_steps % 50000 == 0:
                    self.save(total_steps)

            state = next_state
            if done:
                state = env.reset()
                self.episodes.append(episode_steps)
                episode_steps = 0
        self.replay_memory.reset()
        self.save()

    def plot_training(self, code=None):
        from matplotlib import pyplot as plt
        plt.plot(self.episodes)
        plt.xlabel("Episodes")
        plt.ylabel("Survived steps")
        if code is None:
            code = time.time()
        plt.savefig(f'../saves/{code}.png')
        plt.show()

    def save(self, code=None):
        if code is None:
            code = time.time()
        self.plot_training(code)
        self.target_model.save(f'../saves/{code}.h5')

    def load(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.load_model(path)
        self.epsilon = 0.0

