import numpy as np


class MCTSNode:

    '''
    Implementation of Monte Carlo Tree Search version used in AlphaZero (no rollout)
    
    state arg must be an object inheriting from the GameState base class, with methods:
    1) .legal_actions -> an np.array of indexes of all legal moves
    2) .action() -> accept a move index as argument, apply the move to update the state object, updating all necessary attributes. 
    3) .current_player = player number/label etc. The player whose turn it is, and whose perspective we evaluate the board from
    4) .get_observation() -> A representation of the board from perspective of a specified player
    5) .game_over() -> True or False game terminated
    6) .check_outcome() -> in a game where .game_over() returns True, return the player number/label of the victor OR None in the case of draw

    8) If object has a .display() function, can optionally view the game boards which are explored during search.

    self.prior_function must be able to take in an object of the same type as board_state and return:
    1) a prior probability over ALL moves (not just legal ones) p and a value between 1 and -1
    '''

    _id = 0

    def __init__(self,
        state,
        prior_function,
        c_init=3,
        c_base=19652,
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

        if self.state.game_over:
            self.v = self.state.check_outcome()
            print('Tree reached GAME OVER')

        else:

            # Prior function
            self.prior_function = prior_function

            # Number of legal actions from this state
            self.n_legal_actions = len(self.state.legal_actions)

            # c_init and c_base are used to calculate c_puct, a multiplying factor 
            # on the confidence bound which acts to modulate exploration rate.
            # c_puct slowly increases as number of visits to parent node rises. 
            self.c_init=c_init
            self.c_base=c_base

            # Used to check validity of probability distributions
            self.distribution_validity_epsilon = distribution_validity_epsilon

            # Number of node visits for each child action
            self.N = np.zeros(self.n_legal_actions)

            # Total action values for each child action
            self.W = np.zeros(self.n_legal_actions)

            # Mean action values for each child action
            self.Q = np.zeros(self.n_legal_actions)

            # A list to contain child nodes which are themselves instances of this class, each initialised as None
            self.children = [None]*self.n_legal_actions

            # p contains prior across legal moves in self.legal_moves
            # v contains nnet-estimated value of current state for current player
            #TODO: queue for network evalations
            self.p, self.v = self.prior_function(self.state)
            self.dirichlet_added = False

            # In self-play, at the root node, we add noise for exploration
            if noise_fraction != 0:
                self.add_dirichlet_noise(noise_fraction, dirichlet_alpha)

            # Optionally check that self.p is a valid probability distribution
            if distribution_validity_epsilon:
                self.check_policy_valid()
    

    def add_dirichlet_noise(self, noise_fraction, alpha):
        if self.dirichlet_added:
            raise RuntimeError('Dirichlet noise already added!')
        noise = np.random.dirichlet([alpha]*self.n_legal_actions)
        self.p = (1 - noise_fraction)*self.p + noise_fraction*noise
        self.dirichlet_added = True
        return self.p


    def check_policy_valid(self):
        # Check that self.p is a valid probability distribution
        if np.abs(self.p.sum() - 1.0) > self.distribution_validity_epsilon:
            raise RuntimeError('p is invalid probability distribution: sums to ' + str(self.p.sum()))


    @classmethod
    def new_node(
        cls,
        state,
        prior_function,
        c_init,
        c_base,
        noise_fraction,
        dirichlet_alpha,
        distribution_validity_epsilon
    ):
        return cls(
            state=state,
            prior_function=prior_function,
            c_init=c_init,
            c_base=c_base,
            noise_fraction=noise_fraction,
            dirichlet_alpha=dirichlet_alpha,
            distribution_validity_epsilon=distribution_validity_epsilon
        )


    def U(self):

        '''
        A term added to the action value to encourage exploration

        Larger - more exploration
        '''
        
        N_parent = self.N.sum() + 1

        C = self.C_puct(N_parent)

        return C*self.p*np.sqrt(N_parent)/(1 + self.N)


    def C_puct(self, N_parent):

        '''
        A multiplying factor on the confidence bound, which acts to modulate exploration rate.
        
        Slowly increases as number of visits to parent node rises. 
        '''

        return np.log((1 + N_parent + self.c_base)/self.c_base) + self.c_init


    def action_scores(self):

        '''
        Combine an average action value with a bonus encouraging exploration in line with prior policy
        '''

        # We need to choose moves which maximise value for the current player
        if self.state.current_player == 0:
            player_value = self.Q
        else:
            player_value = -self.Q

        return player_value + self.U()


    def selection(self):

        '''
        Select daughter node for this simulation

        In case that multiple actions have same max score, randomly select from these actions

        (Can make behaviour reproducible with random seed if desired)
        '''

        child_idx = self.action_scores().argmax()

        return child_idx
    

    def expansion(self, child_idx):
        
        # Apply move 
        action = self.state.legal_actions[child_idx]
        new_game_state = self.state.action(action)

        # Create new child node; set noise_fraction to zero - only apply dirichlet noise at root
        new_child = self.new_node(
            new_game_state,
            self.prior_function,
            c_init=self.c_init,
            c_base=self.c_base,
            noise_fraction=0,
            dirichlet_alpha=None,
            distribution_validity_epsilon=self.distribution_validity_epsilon
        )

        self.children[child_idx] = new_child

        return new_child.v


    def backprop(self, idx, v):
        self.N[idx] += 1
        self.W[idx] += v
        self.Q[idx] = self.W[idx]/self.N[idx]


    def simulation(self):

        # Select next node from children of current
        child_idx = self.selection()
        child = self.children[child_idx]

        # If node already present
        if child is not None:

            # If the game is over, return the final value of the game
            if child.state.game_over:
                v = child.v
            
            # Otherwise, continue tree traversal
            else:
                v = child.simulation()

        # If new leaf node, apply expansion. No rollout necessary
        else:
            v = self.expansion(child_idx)

        # Apply backprop at this node
        self.backprop(child_idx, v)

        # Pass down value for backprop in parent node
        return v


    def search_probabilities(self):

        """
        Return the improved policy after MCTS search

        Temperature parameter implemented where this is used (Agent class)
        """
       
        return self.N/self.N.sum()


    def search(self, n_sims=800, msg_every=1e10):

        """Run repeated simulations and return search probabilities"""
        for sim in range(n_sims):

            if sim != 0 and sim % msg_every == 0:
                print(f'Sim {sim}')

            self.simulation()

        return self.search_probabilities()


    def display(self):

        """
        Convenience function - display the internal state of this node
        """

        self.state.display()
        for k, v in vars(self).items():
            print(f'{k}: {v}')        


    def get_node_id(self):
        
        """
        Returns GLOBAL_NODE_IDX by default; override if desired
        """

        return self._id
    

    def get_node_description(self):
        
        """
        Return everything you could possibly want to know about an MCTS node
        """
        boardview = self.state.display()
        N_parent = self.N.sum() + 1

        return {
            'V': self.v,
            'P': self.p.tolist(),
            'N': self.N.tolist(),
            'W': self.W.tolist(),
            'Q': self.Q.tolist(),
            'U': self.U().tolist(),
            'state': boardview,
            'N_parent': N_parent,
            'C_puct': self.C_puct(N_parent),
            'score': self.action_scores().tolist(),
            'current_player': self.state.current_player,
        }


    def get_edge_description(self, child_idx):

        """
        Returns everything you could possibly want to know about an MCTS edge
        """

        if self.children[child_idx] is None:
            raise ValueError('Cannot return edge description to non-existent node')

        return {
            'action': self.state.legal_actions[child_idx],
            'P': self.p[child_idx],
            'N': self.N[child_idx],
            'W': self.W[child_idx],
            'Q': self.Q[child_idx],
            'U': self.U()[child_idx],
            'score': self.action_scores()[child_idx]
        }


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
        edges = {}

        for i, c in enumerate(self.children):
            if c is not None:
                edges[(id, c.get_node_id())] = self.get_edge_description(i)
                c_nodes, c_edges = c.get_nodes_and_edges(to_depth)
                nodes = {**nodes, **c_nodes}
                edges = {**edges, **c_edges}

        return nodes, edges




