import numpy as np
from time import time


class MCTSNode():

    '''
    Base class for a Monte Carlo Tree Search for an adversarial two-player game
    
    state arg must be an object inheriting from the GameState base class, with methods:
    1) .legal_actions -> an np.array of indexes of all legal moves
    2) .action() -> accept a move index as argument, apply the move to update the state object, updating all necessary attributes. 
    3) .current_player = player number/label etc. The player whose turn it is, and whose perspective we evaluate the board from
    4) .get_observation() -> A representation of the board as an array of features
    5) .game_over() -> True or False game terminated
    6) .check_outcome() -> in a game where .game_over() returns True, return the player number/label of the victor OR None in the case of draw

    8) If object has a .display() function, can optionally view the game boards which are explored during search.

    self.prior_function must be able to take in an object of the same type as board_state and return:
    1) a prior probability over ALL moves (not just legal ones) p and a value between 1 and -1

    Make abstract base class later?
    '''

    def __init__(self, state, depth=0, noise_fraction=0, dirichlet_alpha=3, distribution_validity_epsilon=1e-6):

        '''
        Create a new node; 'expansion' step of algorithm
        '''

        # The state of the game at this node
        self.state = state

        # Number of legal actions from this state
        self.n_legal_actions = len(self.state.legal_actions)

        # Keep track of tree depth at this node
        self.depth = depth

        # self.P contains priors p from nnet for all legal moves in self.legal_moves
        # self.V contains nnet-estimated value of current state for current player
        self.p, self.v_init = self.prior_function(self.state.get_observation(), self.state.legal_actions)

        # TODO: In self-play, at the root node, we add noise for exploration
        # if noise_fraction != 0:
        #     noise = dirichlet_noise(dirichlet_alpha, self.n_legal_moves)
        #     self.P = (1 - noise_fraction)*self.P + noise_fraction*noise

        # Check that self.P is a valid probability distribution
        self.distribution_validity_epsilon = distribution_validity_epsilon
        assert np.abs(self.p.sum() - 1.0) < distribution_validity_epsilon, 'self.P is invalid probability distribution: sums to ' + str(self.P.sum())

        # Number of node visits for each child action
        self.N = np.zeros(self.n_legal_actions)

        # Total action values for each child action
        self.W = np.zeros(self.n_legal_actions)

        # Mean action values for each child action
        self.Q = np.zeros(self.n_legal_actions)

        # A list to contain child nodes which are themselves instances of this class, each initialised as None
        self.children = [None]*self.n_legal_actions
    

    def prior_function(self, observation, legal_actions):

        """A dumb prior function for testing.

        Returns:
        
        (i) a policy which is a uniform probability distribution across all the legal moves
        (ii) A value of 0, which reflects an estimate of 50% win probability
        
        """

        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions
        policy = np.ones(n_legal_actions)*uniform_p
        value = 0
        return policy, value


    def rollout(self, lmda):

        '''
        Execute a random rollout from this state to improve our estimate of the state value
        '''

        z = np.random.choice([-1, 1])

        print(f'\n\nROLLOUT OUTCOME: {z} \n\n')
        input()
                      
        self.v_final = (1 - lmda)*self.v_init + lmda*z

        return self.v_final
    

    def selection(self):

        #TODO: what is k value?
        k = 1

        u = k*self.p/(1 + self.N)

        action_scores = self.Q + u

        print('Action scores:', action_scores)
        input()

        return action_scores.argmax()
    

    def backprop(self, idx, v_final):
        self.N[idx] += 1
        self.W[idx] += v_final
        self.Q[idx] = self.W[idx]/self.N[idx]


    def simulation(self, lmda):

        self.display()
        input()

        # Select next node from children of current
        idx = self.selection()
        child = self.children[idx]

        # If node already present, just continue tree traversal
        if child is not None:

            action = self.state.legal_actions[idx]
            print('\nNode already present for action', action)
            input()

            v_final = child.simulation(lmda)


        # If new leaf node, apply expansion and rollout
        else:

            # Expansion: apply move 
            action = self.state.legal_actions[idx]
            new_game_state = self.state.action(action)

            print('\nNo child node for action', action)
            print('Creating new one')
            input()

            # Expansion: create new child node
            new_child = MCTSNode(
                new_game_state,
                depth = self.depth + 1,
                noise_fraction=0,
                dirichlet_alpha=3,
                distribution_validity_epsilon=1e-6
            )

            # Rollout
            v_final = new_child.rollout(lmda)

            self.children[idx] = new_child

        # Apply backprop at this node
        self.backprop(idx, v_final)

        # Pass down value for backprop in parent node
        return v_final



    def display(self):

        """
        Convenience function - display the internal state of this node
        """

        self.state.display()
        for k, v in vars(self).items():
            print(f'{k}: {v}')


    # def update(self):

    #     '''
    #     Execute a single update of the tree
    #     '''

    #     # Number of visits to THIS node is sum of visits to all child nodes
    #     N_parent = np.sum(self.N)

    #     # A factor to modulate exploration rate
    #     C_puct = self.C_puct(N_parent)

    #     # Upper confidence bound for each child action
    #     self.U = C_puct * self.P * np.sqrt(N_parent) / (1 + self.N) # elementwise division
    #     self.upper_bound = self.Q + self.U 

    #     # Choose action based on upper confidence bound
    #     chosen_child_index = np.idxmax(self.upper_bound)

    #     # If the chosen child is a leaf node, create a new node there 
    #     if self.children[chosen_child_index] is None:

    #         # Get the move that this child corresponds to
    #         chosen_move_id = self.legal_moves[chosen_child_index]

    #         # Make a copy of the state and apply the chosen move to the copy
    #         new_state = self.state.__copy__().move(chosen_move_id)

    #         # If the new state is a terminated game:
    #         if new_state.game_over():
                
    #             # Who won? Get the player number/label
    #             winner = new_state.check_winner()
                
    #             # Value is 1 if the current player has won, 0 for draw, -1 for loss
    #             value_to_current_node = 1 if (winner == self.state.current_player) else (0 if winner is None else -1)

    #             # There is no need to create a node for the terminated game. The visit count for this outcome and it's value are stored in the current node.

    #         # If there are still moves to play from the leaf node:
    #         else:
    #             # Expand the leaf node by creating a new instance of this class for the chosen move.
    #             # Child nodes created in this way will never have noise, so leave noise_fraction at default 0
    #             self.children[chosen_child_index] = MCTS(new_state, distribution_validity_epsilon=self.distribution_validity_epsilon)

    #             # To see what the value of this new state is to the current node, first see who the player is in the new state.
    #             same_player = new_state.current_player == self.state.current_player

    #             # If it's the other player, the value needs a sign change: this implementation always evaluates each position from the perspective of the next player to play. 
    #             # A good value for the other player is a bad value for the current player. 
    #             value_to_current_node = (1 if same_player else -1)*self.children[chosen_child_index].V
        
    #     # If it's not a leaf node - i.e it's already expanded - choose it and run this function within it to continue the simulation
    #     else:
    #         value_to_child_node = self.children[chosen_child_index].update()

    #         # Again if the chosen child node has a different current_player, we must reverse the value
    #         same_player = self.children[chosen_child_index].state.current_player == self.state.current_player
    #         value_to_current_node = (1 if same_player else -1)*value_to_child_node

    #     # Update visit count for chosen child
    #     self.N[chosen_child_index] += 1

    #     # Update total value for chosen child 
    #     self.W[chosen_child_index] += value_to_current_node        

    #     # Update action value for this action using new value information. 
    #     # Action value is just mean of returned values - so could use iterative average update... but this is simpler
    #     self.Q[chosen_child_index] = self.W[chosen_child_index]/self.N[chosen_child_index]

    #     return value_to_current_node


    # def get_policy(self, temperature):

    #     return self.N**(1/temperature) # raise to power elementwise


    # @abstractmethod
    # def prior_function(self, state_representation):

    #     '''take in board, output vector p and scalar v'''

    #     pass


    # def C_puct(self, N_parent):

    #     '''
    #     A multiplying factor on the confidence bound, which acts to modulate exploration rate.
        
    #     Slowly increases as number of visits to parent node rises. 
    #     '''

    #     c_init = 1.25
    #     c_base = 19652

    #     C = np.log((1 + N_parent + c_base)/c_base) + c_init

    #     return C


    # def build_tree_and_choose_move(self, time_limit=None, sims_limit=None):

    #     '''
    #     Repeatedly run simulations to update the MCTS tree. Termination based on time expired OR number of simulations. 

    #     At the end, return the move with the best average action value and its associated MCTS tree - this can be used to save some time
    #     during self play (already partially-explored tree!)
    #     '''

    #     assert time_limit or sims_limit, 'Must use at least one of time_limit, sims_limit'

    #     if not time_limit:
    #         time_limit = np.inf
    #     if not sims_limit:
    #         sims_limit = np.inf

    #     sim_count = 0
    #     start_time = time()

    #     while time() - start < time_limit and sim_count < sims_limit:
    #         self.update()
    #         sim_count += 1

    #     best_child = np.idxmax(self.N)
    #     best_move = self.legal_moves[best_child]

    #     return best_move, self.children[best_child]



def dirichlet_noise(alpha, n_legal_moves):
    positive_concentration_params = [alpha]*n_legal_moves
    noise = np.random.dirichlet(positive_concentration_params)
    return noise

