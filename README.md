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

