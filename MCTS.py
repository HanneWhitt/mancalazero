import numpy as np
from time import time



class MCTS:

    '''
    Base class for MCTS. 
    
    state arg must be an object with the following attributes/methods:
    1) .get_legal_moves() -> an np.array of indexes of all legal moves
    2) .move() -> accept a move index as argument, apply the move to update the state object, updating all necessary attributes. 
    3) .current_player = player number/label etc. The player whose turn it is, and whose perspective we evaluate the board from
    4) .get_representation() -> A representation of the board as an array of features
    5) .game_over() -> True or False game terminated
    6) .check_winner() -> in a game where .game_over() returns True, return the player number/label of the victor OR None in the case of draw
    7) .__copy__() -> a copy of the state object which can be updated without changing the original

    8) If object has a .display() function, can optionally view the game boards which are explored during search.

    self.prior_function must be able to take in an object of the same type as board_state and return:
    1) a prior probability over ALL moves (not just legal ones) p and a value between 1 and -1
    '''

    def __init__(state, noise_fraction=0, dirichlet_alpha=3, distribution_validity_epsilon=1e-6):

        '''
        Create a new node
        '''

        # A copy of the root board state for this MCTS tree
        self.state = state

        # Get legal moves from state (a list of indices)
        self.legal_moves = self.state.get_legal_moves()
        self.n_legal_moves = len(self.legal_moves)

        # self.P contains priors p from nnet for all legal moves in self.legal_moves
        # self.V contains nnet-estimated value of current state for current player
        self.state_representation = self.state.get_representation()
        self.P, self.V = self.prior_function(self.state_representation, self.legal_moves)

        # In self-play, at the root node, we add noise for exploration
        if noise_fraction != 0:
            noise = dirichlet_noise(dirichlet_alpha, self.n_legal_moves)
            self.P = (1 - noise_fraction)*self.P + noise_fraction*noise

        # Check that self.P is a valid probability distribution
        self.distribution_validity_epsilon = distribution_validity_epsilon
        assert np.abs(self.P.sum() - 1.0) < distribution_validity_epsilon, 'self.P is invalid probability distribution: sums to ' + str(self.P.sum())

        # Total action values for each child action
        self.W = np.zeros(self.n_legal_moves)

        # Mean action values for each child action
        self.Q = np.zeros(self.n_legal_moves)
        
        # Number of node visits for each child action
        self.N = np.zeros(self.n_legal_moves)

        # A list to contain child nodes which are themselves instances of this class, each initialised as None
        self.children = [None]*self.n_legal_moves



    def update(self):

        '''
        Execute a single update of the tree
        '''

        # Number of visits to THIS node is sum of visits to all child nodes
        N_parent = np.sum(self.N)

        # A factor to modulate exploration rate
        C_puct = self.C_puct(N_parent)

        # Upper confidence bound for each child action
        self.U = C_puct * self.P * np.sqrt(N_parent) / (1 + self.N) # elementwise division
        self.upper_bound = self.Q + self.U 

        # Choose action based on upper confidence bound
        chosen_child_index = np.idxmax(self.upper_bound)

        # If the chosen child is a leaf node, create a new node there 
        if self.children[chosen_child_index] is None:

            # Get the move that this child corresponds to
            chosen_move_id = self.legal_moves[chosen_child_index]

            # Make a copy of the state and apply the chosen move to the copy
            new_state = self.state.__copy__().move(chosen_move_id)

            # If the new state is a terminated game:
            if new_state.game_over():
                
                # Who won? Get the player number/label
                winner = new_state.check_winner()
                
                # Value is 1 if the current player has won, 0 for draw, -1 for loss
                value_to_current_node = 1 if (winner == self.state.current_player) else (0 if winner is None else -1)

                # There is no need to create a node for the terminated game. The visit count for this outcome and it's value are stored in the current node.

            # If there are still moves to play from the leaf node:
            else:
                # Expand the leaf node by creating a new instance of this class for the chosen move.
                # Child nodes created in this way will never have noise, so leave noise_fraction at default 0
                self.children[chosen_child_index] = MCTS(new_state, distribution_validity_epsilon=self.distribution_validity_epsilon)

                # To see what the value of this new state is to the current node, first see who the player is in the new state.
                same_player = new_state.current_player == self.state.current_player

                # If it's the other player, the value needs a sign change: this implementation always evaluates each position from the perspective of the next player to play. 
                # A good value for the other player is a bad value for the current player. 
                value_to_current_node = (1 if same_player else -1)*self.children[chosen_child_index].V
        
        # If it's not a leaf node - i.e it's already expanded - choose it and run this function within it to continue the simulation
        else:
            value_to_child_node = self.children[chosen_child_index].update()

            # Again if the chosen child node has a different current_player, we must reverse the value
            same_player = self.children[chosen_child_index].state.current_player == self.state.current_player
            value_to_current_node = (1 if same_player else -1)*value_to_child_node

        # Update visit count for chosen child
        self.N[chosen_child_index] += 1

        # Update total value for chosen child 
        self.W[chosen_child_index] += value_to_current_node        

        # Update action value for this action using new value information. 
        # Action value is just mean of returned values - so could use iterative average update... but this is simpler
        self.Q[chosen_child_index] = self.W[chosen_child_index]/self.N[chosen_child_index]

        return value_to_current_node


    def get_policy(self, temperature):

        return self.N**(1/temperature) # raise to power elementwise


    def prior_function(self, state_representation):

        '''take in board, output vector p and scalar v'''

        raise NotImplementedError


    def C_puct(self, N_parent):

        '''
        A multiplying factor on the confidence bound, which acts to modulate exploration rate.
        
        Slowly increases as number of visits to parent node rises. 
        '''

        c_init = 1.25
        c_base = 19652

        C = np.log((1 + N_parent + c_base)/c_base) + c_init

        return C


    def build_tree_and_choose_move(self, time_limit=None, sims_limit=None):

        '''
        Repeatedly run simulations to update the MCTS tree. Termination based on time expired OR number of simulations. 

        At the end, return the move with the best average action value and its associated MCTS tree - this can be used to save some time
        during self play (already partially-explored tree!)
        '''

        assert time_limit or sims_limit, 'Must use at least one of time_limit, sims_limit'

        if not time_limit:
            time_limit = np.inf
        if not sims_limit:
            sims_limit = np.inf

        sim_count = 0
        start_time = time()

        while time() - start < time_limit and sim_count < sims_limit:
            self.update()
            sim_count += 1

        best_child = np.idxmax(self.N)
        best_move = self.legal_moves[best_child]

        return best_move, self.children[best_child]



def dirichlet_noise(alpha, n_legal_moves):
    positive_concentration_params = [alpha]*n_legal_moves
    noise = np.random.dirichlet(positive_concentration_params)
    return noise




# class MancalaPlayer

# return np.ones(5)*(1/5), 0.5

if __name__ == "__main__":


    # testing 


    def C_puct(N_parent, c_init=1.25, c_base=19652):

        '''
        A multiplying factor on the confidence bound.
        '''

        C = np.log((1 + N_parent + c_base)/c_base) + c_init

        return C

    N = np.array([0, 1, 0, 2, 0, 3])
    P = np.array([0, 0.1, 0.3, 0.4, 0, 0.2])
    C_puct = 1.25
    N_parent = 6

    U = C_puct * P * np.sqrt(N_parent) / (1 + N)

    print(U)

    noise = dirichlet_noise(3, 3)

    print('\n\n\n', noise, np.sum(noise))