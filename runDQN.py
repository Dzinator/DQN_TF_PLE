from ple.games.pixelcopter import Pixelcopter
#from ple.games.pong import Pong
from ple import PLE
from tqdm import tqdm
from deepQ import DQN
import numpy as np
import cv2


def preprocess(observation):
    # observation = cv2.resize(observation, (84, 100)) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
    # observation = observation[26:110, :]
    # ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84,84,1))
    # print(observation.shape)
    # return observation

def playGame():
    # 1 - init game
    game = Pixelcopter(height=84, width=84)
    #game = Pong(height=84, width=84)
    p = PLE(game, fps=60, display_screen=True, reward_values={
        "positive": 2.0,
        "tick": 0.1
    })
    p.init()

    # 2 - init DQN agent
    legal_actions = p.getActionSet()
    agent = DQN(actions=len(legal_actions))


    # 3 - play game
    init_action = np.array([1, 0, 0, 0])
    init_observation = p.getScreenGrayscale()
    init_reward = p.score()
    init_terminal = p.game_over()

    agent.setInitState(init_observation)


    while True:
        if p.game_over():
            p.reset_game()

        action_list = agent.getAction()
        action = np.argmax(action_list)  # pick non-zero valued action index
        p.act(legal_actions[action])
        reward = p.score() #p.act(action)
        # print(reward)
        observation = preprocess(p.getScreenGrayscale())
        terminal = p.game_over()

        agent.setPerception(observation, action_list, reward, terminal)


def main():
    playGame()

if __name__ == '__main__':
    main()