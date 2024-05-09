# mancalazero
An implementation of AlphaZero for the ancient African game, Mancala!

Part complete. Implemented using the methods described in: 

"[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://science.sciencemag.org/content/362/6419/1140.full?ijkey=XGd77kI6W4rSc&keytype=ref&siteid=sci)", Silver et al., DeepMind, 2018

"[Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)", Silver et al., DeepMind, 2017

The implementation is from the ground up, with fully original code and the only non-standard Python packages used being numpy, torch, and networkx/graphviz for MCTS tree visualisation. 

### Scripts:

* ```gamestate.py``` contains an abstract base class (ABC) ```GameState``` intended as a framework for the implementation of a wide range of classical games; I hope that this strategy will make it possible to extend this project quite simply to other games once it works for Mancala. A simple test game, involving flipping a coin, can be seen in ```tests/gamestate_test.py```; ```mancala.py``` implements the rules of Mancala in several variations, and allows access to the state of a game as a vector of length 16, with features for the number of stones in each of the 14 spaces on the board, the current player, and the current turn number. 

* ```mancalanet.py``` uses PyTorch to build a neural network suitable for use in AlphaZero, taking as input the state of a game, and outputing a policy vector and value estimate. The policy vector is a probability distribution describing the networks' view of which moves are more or less promising, and the value provides the networks' estimate of win probability from the current position. The AlphaZero loss function is implemented in ```loss.py```, and uses the outcomes of real games to train the value component, the MCTS search probabilities to train the policy component, and a customised L2 weight regularisation as a measure against overfitting. 

* ```MCTS.py``` implements Monte Carlo Tree Search, an algorithm which extends the neural network's strength by using it to look ahead and intelligently sample possible outcomes resulting from different moves. The version used in AlphaZero, unusually, does not include a rollout, instead using the neural network alone to estimate value. ```MCTSNode``` therefore implements just the selection, expansion and backpropagation steps for each simulation, as well as the other small modifications made in AlphaZero, such as the addition of Dirichlet noise to the policy at the root node. ```visualisation.py``` provides functions to display the MCTS graph, and the data associated with each node and edge, as the tree expands. 

* ```agent.py``` implements a very simple ABC ```Agent``` for agents which play games, requiring only a 'policy' method. Inheriting from this, ```RandomAgent``` chooses moves using a random uniform distribution, used in tests and during training initialisation. ```AlphaZeroAgent``` implements the final agent, using the network, and returning the search probabilities from MCTS as its policy.

* ```selfplay.py```, plays ```Agent```s against each other to generate games that can be used to train the network, with options for random sampling of moves and control over the temperature parameter used for selection of moves depending on move number (allows tuning for exploratory vs. optimal play). This is currently under development; I hope to use an asynchronous producer-consumer pattern to maintain a large buffer of games that a torch ```DataLoader``` can randomly sample from, imitating the approach used in the original research. 


### Notes

##### Choice of rules variant
The version I know from childhood:

* starts with 3 stones in every hole; 
* ends when a player is out of stones at the beginning of his turn and cannot make a legal move - all the stones remaining in play are awarded to the other player.

This is one of the options implemented in `mancala.py` (`starting_stones=3, no_moves_policy='end_game'`). 

However, upon running a large number of games between agents that randomly select their moves, it became clear that this version gives a considerable advantage to the first player to move. 

I'd ideally like to develop an agent to play a relatively balanced version of the game, and I wondered if other variants might be a bit more even. I have also previously come across versions which start with 4 stones, and which don't end upon one player running out of stones, instead simply passing play back to the other player (`no_moves_policy='pass_back'`). Both changes seem likely to result in longer games, perhaps watering down the advantage of taking the first move?

Upon running 100,000 games of each version between random agents, we can see the impact of these changes. Both increasing the number of moves and switching to the `pass_back` variant result, as expected, in longer games. However, only increasing the number of stones improves balance. 

The version with 4 stones and the `end_game` variant has the smallest difference in win percentage between Player 0 and Player 1, and so this is the version I'll take forward. 


| Variant | P0 wins (%) | P1 wins (%) | P0 - P1 (%) | Draws (%) | P0 average score | P1 average score | Average turns played |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| end_game_variant_3_stones | 49.3 | 42.8 | 6.4 | 7.9 | 18.4 | 17.6 | 31.8 |
| pass_back_variant_3_stones | 48.8 | 41.4 | 7.4 | 9.8 | 18.3 | 17.7 | 37.2 |
| end_game_variant_4_stones | 48.4 | 44.8 | **3.6** | 6.7 | 24.2 | 23.8 | 42.4 |
| pass_back_variant_4_stones | 48.0 | 44.1 | 3.9 | 7.9 | 24.2 | 23.8 | 47.8 |

