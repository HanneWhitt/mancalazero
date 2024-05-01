from mancalazero.Agent import Agent, RandomAgent
from collections import Counter
import numpy as np


random_agent = RandomAgent()

legal_actions = [0, 1, 3, 5]

p = random_agent.policy(None, legal_actions)

print(p)
input()

a = random_agent.select_action(None, legal_actions)

print(a)

choices = [random_agent.select_action(None, legal_actions)[0] for n in range(100000)]

print(Counter(choices))



class FixedAgent(Agent):
    def policy(self, state, legal_actions):
        return np.array([0.2, 0, 0.7, 0.1])
    

legal_actions = ['a', 'b', 'c', 'd']

fixed_agent = FixedAgent()

print('TEMP 1')
choices = [fixed_agent.select_action(None, legal_actions, temperature=1)[0] for n in range(100000)]
print(Counter(choices))

print('TEMP 0')
choices = [fixed_agent.select_action(None, legal_actions, temperature=0)[0] for n in range(100000)]
print(Counter(choices))

print('TEMP 10')
choices = [fixed_agent.select_action(None, legal_actions, temperature=10) for n in range(100000)]
print(Counter(choices))