#
#      Mathieu Boucher and Yanis Hattab
#
#          DeepMind Nature's publication Deep Q-learning implementation with PyGame (Windows friendly)
#

from collections import deque
import tensorflow as tf
import numpy as np
import random

#Training hyper parameters / initialized based on paper

FRAME_PER_ACTION = 1
GAMMA = 0.95 #decay rate for Q
OBSERVE_SET = 5000#0
EXPLORE_SET = 10000#00
INIT_EPSILON = 1.0
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

        self.timeStep = 0
        self.epsilon = INIT_EPSILON
        self.actions = actions

        #init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3,\
            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

        #init target Q network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T,\
            self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        #copy operation
        self.copyTargetNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                           self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                           self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                           self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                           self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        #saving and loading
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("./saved/")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Couldn't find old network")

    def createQNetwork(self):
        #init weights and biases for all layers, based on architecture in paper
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([3136, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        stateInput = tf.placeholder("float", [None, 84, 84, 4]) #input layer

        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_shape = h_conv3.get_shape().as_list()

        print("Dimension:", h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])

        h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        qValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, qValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    def copyTargetNetwork(self):
        self.session.run(self.copyTargetNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])

        self.yInput = tf.placeholder("float", [None])

        q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)

        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.square(self.yInput - q_action))

        tf.summary.scalar("Cost", self.cost)

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
            self.yInput : y_batch,
            self.actionInput : action_epoch,
            self.stateInput : state_epoch
        })

        #save every 10000 iterations TODO
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, './saved/network-dqn', global_step=self.timeStep)

        # tensor board logs
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs/", self.session.graph)

        if self.timeStep % UPDATE_TIME == 0:
            # copy NN
            self.copyTargetNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.append(nextObservation, self.currentState[:, :, :-1], axis=2)

        self.replayMemory.append((self.currentState, action, reward, newState, terminal))

        tf.summary.scalar("Score", reward)

        if self.timeStep > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE_SET:
            #train
            self.train()

        #print status
        state = ""
        if self.timeStep <= OBSERVE_SET:
            state = "observe"
        elif self.timeStep > OBSERVE_SET and self.timeStep <= OBSERVE_SET + EXPLORE_SET:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP ", self.timeStep, "/ STATE ", state, "/ EPSILON ", self.epsilon)
        tf.summary.scalar("Epsilon", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        qValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        action_i = 0

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

    def weight_variable(self, shape):
        init = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init)

    def bias_variable(self, shape):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")





