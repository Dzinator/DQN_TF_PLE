#
#      Mathieu Boucher and Yanis Hattab
#
#      DeepMind Nature's publication Deep Q-learning implementation with PyGame (Windows friendly)
#

from collections import deque
import tensorflow as tf
import numpy as np
import random

#Training hyper parameters / initialized based on paper
FRAME_PER_ACTION = 1
GAMMA = 0.95 #decay rate for Q
OBSERVE_SET = 50000
EXPLORE_SET = 1000000
INIT_EPSILON = 0.3#1.0
FINAL_EPSILON = 0.1
REPLAY_MEMORY = 1000000 #previous transitions to store
NB_EPOCHS = 32
UPDATE_TIME = 10000

LEARNING_RATE = 0.00025
DECAY_RATE = 0.99


class DQN:
    """Class implementing the Deep Q-Learning Algorithm"""

    def __init__(self, actions):


        self.replayMemory = deque()
        self.rewards = []
        self.rewards_ph = tf.placeholder(tf.float32, name='rewards')
        self.timeStep = 0
        self.epsilon = INIT_EPSILON
        self.epsilon_ph = tf.placeholder(tf.float32, name='epsilon')
        tf.summary.scalar("Epsilon", self.epsilon_ph)

        self.actions = actions

        #init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3,\
            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

        #init target Q network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T,\
            self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        # logger with graph




        #copy operation
        self.copyTargetNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                           self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                           self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                           self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                           self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        #saving, logging and loading
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.logger = tf.summary.FileWriter("./logs/", self.session.graph)
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("./saved/")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Couldn't find old network")

    def createQNetwork(self):
        #init weights and biases for all layers, based on architecture in paper
        with tf.name_scope("Weights"):
            W_conv1 = self.weight_variable([8, 8, 4, 32], "W_conv1")
            W_conv2 = self.weight_variable([4, 4, 32, 64], "W_conv2")
            W_conv3 = self.weight_variable([3, 3, 64, 64], "W_conv3")

            W_fc1 = self.weight_variable([3136, 512], "W_fullCon1")
            W_fc2 = self.weight_variable([512, self.actions], "W_fullCon2")

        with tf.name_scope("Biases"):
            b_conv1 = self.bias_variable([32], "B_conv1")
            b_conv2 = self.bias_variable([64], "B_conv2")
            b_conv3 = self.bias_variable([64], "B_conv3")

            b_fc1 = self.bias_variable([512], "B_fullCon1")
            b_fc2 = self.bias_variable([self.actions], "B_fullCon2")

        with tf.name_scope("Input_layer"):
            stateInput = tf.placeholder("float", [None, 84, 84, 4], name="inputs") #input layer

        with tf.name_scope("Hidden_layers"):
            h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1, name="h_conv1")
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2, name="h_conv2")
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3, name="h_conv3")

            h_conv3_shape = h_conv3.get_shape().as_list()

            print("Dimension:", h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])

            h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, "h_fc1")

        qValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, qValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    def copyTargetNetwork(self):
        self.session.run(self.copyTargetNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])

        self.yInput = tf.placeholder("float", [None])

        q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.yInput - q_action))

        tf.summary.scalar("Cost", self.cost)

        tf.summary.scalar("Average_reward", tf.reduce_mean(self.rewards_ph))

        self.trainStep = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, 0.0, 1e-6).minimize(self.cost)


    def train(self):

        # 1) obtain random mini-batch (epoch) from replay memory
        epoch = random.sample(self.replayMemory, NB_EPOCHS)
        state_epoch = [data[0] for data in epoch]
        state_epoch = (np.array(state_epoch))
        action_epoch = [data[1] for data in epoch]
        reward_epoch = [data[2] for data in epoch]
        # print(np.array(action_epoch).shape)
        nextState_epoch = [data[3] for data in epoch]

        # 2) calculate ideal action y
        y_batch = []
        qValue_epoch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_epoch})
        for i in range(0, NB_EPOCHS):
            terminal = epoch[i][4]
            if terminal:
                y_batch.append(reward_epoch[i])
            else:
                y_batch.append(reward_epoch[i] + GAMMA * np.max(qValue_epoch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_epoch,
            self.stateInput: state_epoch
        })

        if self.timeStep % 1000 == 0:
            # log summary
            merged = tf.summary.merge_all()
            summary = self.session.run(merged, feed_dict={
                self.yInput: y_batch,
                self.actionInput: action_epoch,
                self.stateInput: state_epoch,
                self.rewards_ph : self.rewards,
                self.epsilon_ph : self.epsilon
            })
            self.logger.add_summary(summary, self.timeStep)

        #save every 10000 iterations
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, './saved/dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            # copy NN
            self.copyTargetNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.append(nextObservation, self.currentState[:, :, 1:], axis=2)

        self.replayMemory.append((self.currentState, action, reward, newState, terminal))

        if self.timeStep > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE_SET:
            self.train()

        #save rewards pool for logging
        self.rewards.append(reward)
        if len(self.rewards) > REPLAY_MEMORY:
            self.rewards = self.rewards[1:]

        #print status
        state = ""
        if self.timeStep <= OBSERVE_SET:
            state = "observe"
        elif self.timeStep > OBSERVE_SET and self.timeStep <= OBSERVE_SET + EXPLORE_SET:
            state = "explore"
        else:
            state = "train"

        #console logging TODO
        # print("TIMESTEP ", self.timeStep, "/ STATE ", state, "/ EPSILON ", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        qValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)

        if self.timeStep % FRAME_PER_ACTION == 0:
            #choose an action if enough time has elapsed
            if random.random() <= self.epsilon:
                action_i = random.randrange(self.actions)
                action[action_i] = 1
            else:
                action_i = np.argmax(qValue)
                action[action_i] = 1
        else:
            action[0] = 1 #don't act

        #update epsilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE_SET:
            self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / EXPLORE_SET

        # print("ACTION:", action)
        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape, name):
        init = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init, name)

    def bias_variable(self, shape, name):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init, name)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")





