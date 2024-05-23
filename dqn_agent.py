import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory      = []
        self.max_memory_size = 10000
        self.gamma           = 0.99
        self.epsilon         = 1.0
        self.epsilon_min     = 0.01
        self.epsilon_decay   = 0.995
        self.batch_size      = 64
        self.model = self._build_model()
        self.rewards_history = []
        self.loss_history    = []

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_actions, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([np.squeeze(experience[0]) for experience in minibatch])

        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        next_states = np.reshape(next_states, (self.batch_size, self.state_shape[0]))

        targets = self.model.predict(states)
        next_state_targets = rewards + (1 - dones) * self.gamma * np.amax(self.model.predict(next_states), axis=1)
        targets[np.arange(self.batch_size), actions] = next_state_targets

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay