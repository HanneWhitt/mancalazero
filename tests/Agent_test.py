from mancalazero.agent import Agent, RandomAgent
from mancalazero.mancala import Mancala
from collections import Counter
import numpy as np

state = Mancala(starting_stones=4)

random_agent = RandomAgent()

# legal_actions = [0, 1, 3, 5]

p = random_agent.policy(state)

print(p)
input()

a = random_agent.select_action(state)

print(a)

choices = [random_agent.select_action(state)[0] for n in range(100000)]

print(Counter(choices))



class FixedAgent(Agent):
    def policy(self, state):
        return np.array([0.2, 0, 0.5, 0.1, 0.1, 0.1])
    

state.legal_actions = ['a', 'b', 'c', 'd', 'e', 'f']

fixed_agent = FixedAgent()

print('TEMP 1')
choices = [fixed_agent.select_action(state, temperature=1)[0] for n in range(100000)]
print(Counter(choices))

print('TEMP 0')
choices = [fixed_agent.select_action(state, temperature=0)[0] for n in range(100000)]
print(Counter(choices))

print('TEMP 10')
choices = [fixed_agent.select_action(state, temperature=10) for n in range(100000)]
print(Counter(choices))