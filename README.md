# mancalazero
An implementation of AlphaZero for the ancient African game, Mancala!

Part complete. Implemented using the methods described in: 

"[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://science.sciencemag.org/content/362/6419/1140.full?ijkey=XGd77kI6W4rSc&keytype=ref&siteid=sci)", Silver et al., DeepMind, 2018

"[Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)", Silver et al., DeepMind, 2017

The implementation is from the ground up, with fully original code and the only non-standard python libraries used being numpy and torch.

### Scripts:

* ```mancala.py``` implements the rules of mancala in several variations.

* ```policy_and_value_network.py``` implements a neural network which estimates, from the board position, (i) the value of the game to the current player (related to win probability), (ii) the policy - i.e a probability distribution with higher values for good moves. 

* ```MCTS.py``` implements Monte Carlo Tree Search, an algorithm which extends the neural network's strength by using it to look ahead and intelligently sample possible outcomes resulting from different moves. 

* ```self_play.py```, (part complete) which plays off MCTS agents against each other to generate games that can be used to train the network. 
