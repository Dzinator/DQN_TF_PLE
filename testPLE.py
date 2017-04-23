from ple.games.pixelcopter import Pixelcopter
from ple import PLE
from tqdm import tqdm
import random
import pygame

class Aigent(object):
    actions = []

    def __init__(self, allowed_actions):
        self.actions = allowed_actions
        print(self.actions)

    def pickAction(self, reward, observation):
        # print(len(self.actions))
        if bool(random.getrandbits(1)):
            return self.actions[0]
        else:
            return self.actions[0]


game = Pixelcopter(height=80, width=80)
p = PLE(game, fps=30, display_screen=True)
agent = Aigent(allowed_actions=p.getActionSet())

p.init()
reward = 0.0

for i in tqdm(range(50000)):
    if p.game_over():
        p.reset_game()

    observation = p.getScreenGrayscale()

    action = agent.pickAction(reward, observation)
    print(action)
    reward = p.act(action)

