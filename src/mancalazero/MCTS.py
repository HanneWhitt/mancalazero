import numpy as np
from abc import ABC, abstractmethod


class MCTSNode(ABC):

    '''
    Implementation of Monte Carlo Tree Search version used in AlphaZero (no rollout)
    
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
    '''

    _id = 0

    def __init__(self,
        state,
        noise_fraction=0,
        dirichlet_alpha=3,
        distribution_validity_epsilon=1e-6
    ):

        '''
        Create a new node; 'expansion' step of algorithm
        '''

        # Unique id for this node
        self._id = MCTSNode._id
        MCTSNode._id += 1

        # The state of the game at this node
        self.state = state

        # Number of legal actions from this state
        self.n_legal_actions = len(self.state.legal_actions)

        # self.P contains priors p from nnet for all legal moves in self.legal_moves
        # self.V contains nnet-estimated value of current state for current player
        self.p, self.v = self.prior_function(self.state.get_observation(), self.state.legal_actions)

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


    @abstractmethod
    def prior_function(self, observation, legal_actions):

        """
        Implement a function that returns:
         
        (i) a policy vector with the same length as the number of legal actions
        (ii) a value between -1 and 1
        
        """

        pass
    

    def get_node_id(self):
        
        """
        Returns GLOBAL_NODE_IDX by default; override if desired
        """

        return self._id
    

    def get_node_description(self):
        
        """
        Returns observation, P, V, N, W and Q by default; override if desired
        """

        return {
            'P': self.p.tolist(),
            'V': self.v,
            'N': self.N.tolist(),
            'W': self.W.tolist(),
            'Q': self.Q.tolist()
        }


    @classmethod
    def new_node(
        cls,
        state,
        noise_fraction=0,
        dirichlet_alpha=3,
        distribution_validity_epsilon=1e-6
    ):
        return cls(
            state,
            noise_fraction=noise_fraction,
            dirichlet_alpha=dirichlet_alpha,
            distribution_validity_epsilon=distribution_validity_epsilon
        )


    def selection(self):

        #TODO: what is k value?
        k = 1

        u = k*self.p/(1 + self.N)

        action_scores = self.Q + u

        return action_scores.argmax()
    

    def backprop(self, idx, v):
        self.N[idx] += 1
        self.W[idx] += v
        self.Q[idx] = self.W[idx]/self.N[idx]


    def simulation(self):

        # Select next node from children of current
        idx = self.selection()
        child = self.children[idx]

        # If node already present, just continue tree traversal
        if child is not None:

            v = child.simulation()

        # If new leaf node, apply expansion. No rollout necessary
        else:

            # Expansion: apply move 
            action = self.state.legal_actions[idx]
            new_game_state = self.state.action(action)

            # Expansion: create new child node
            new_child = self.new_node(
                new_game_state,
                noise_fraction=0,
                dirichlet_alpha=3,
                distribution_validity_epsilon=1e-6
            )

            self.children[idx] = new_child

            v = new_child.v

        # Apply backprop at this node
        self.backprop(idx, v)

        # Pass down value for backprop in parent node
        return v


    def search_probabilities(self, temperature=1):

        """
        Return the improved policy after MCTS search, modulated by temp parameter
        """

        pi = self.N**(1/temperature)

        # standardise
        pi = pi/pi.sum()

        return pi


    def display(self):

        """
        Convenience function - display the internal state of this node
        """

        self.state.display()
        for k, v in vars(self).items():
            print(f'{k}: {v}')        


    def get_nested_dict(self, to_depth=None):

        """
        Compile a nested dict representation of the (sub)tree starting at this node
        """

        if to_depth is not None:
            if to_depth < 0:
                return None
            to_depth -= 1

        children = [c if c is None else c.get_tree_description(to_depth) for c in self.children]

        tree_description = {
            'id': self.get_node_id(),
            'description': self.get_node_description(),
            'children': children
        }

        return tree_description


    def get_nodes_and_edges(self, to_depth=None):

        """
        Compile a list of nodes and edges of the (sub)tree starting at this node
        """

        if to_depth is not None:
            if to_depth < 0:
                return {}, []
            to_depth -= 1

        id = self.get_node_id()

        nodes = {id: self.get_node_description()}
        edges = []

        for c in self.children:
            if c is not None:
                edges.append((id, c.get_node_id()))
                c_nodes, c_edges = c.get_nodes_and_edges(to_depth)
                nodes = {**nodes, **c_nodes}
                edges = edges + c_edges

        return nodes, edges


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

