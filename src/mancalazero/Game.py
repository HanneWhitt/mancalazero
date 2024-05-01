from abc import ABC, abstractmethod


class GameState(ABC):
    
    '''
    Abstract base class for classical games, exposing functions required for AlphaZero algorithm

    Essentially a state with functions attached
    '''


    def __init__(self, state=None, move_history=True, check_validity=True):
        if state is None:
            state = self.new_game_state()
        self.check_validity = check_validity
        self.state = state
        if move_history:
            if isinstance(move_history, list):
                self.move_history = move_history.copy()
            else:
                self.move_history = []
        else:
            self.move_history = False


    @property
    @abstractmethod
    def actions_list(self):
        '''
        Should return a list of all possible actions in the game
        '''
        pass


    @classmethod 
    def total_actions(cls):
        return len(cls.actions_list)


    @abstractmethod
    def new_game_state(self):
        '''
        Return the state of a new game
        '''
        pass


    @property
    def state(self):
        return self._state
    

    @state.setter
    def state(self, state):
        '''
        Whenever the state is updated:

        i) Check that the new state is valid under the rules
        ii) Check if the new state describes a finished game
        iii) Update list of legal actions to match this state
        '''
        self._state = state
        if self.check_validity:
            self.check_state_validity()
        self.game_over = self.is_game_over()
        if self.game_over:
            self.legal_actions = None
        else:
            self.legal_actions = self.get_legal_actions()


    @property
    @abstractmethod
    def current_player(self):
        '''
        Return the current player index. Just return 0 for single-player games, 0 or 1 for 2 player, etc.
        '''
        pass


    @property
    @abstractmethod
    def turn_number(self):
        '''
        Return the current turn number.
        '''
        pass


    def get_observation(self, player):
        '''
        Return a tensor representing an observation of the state from the point of view of a specific player.

        Defaults to returning the state itself.
        
        Can be overriden in games where observation is different to state.
        '''
        return self.state


    @abstractmethod
    def check_state_validity(self):
        '''
        Run a check that the current configuration of the game is permitted by the rules
        '''
        pass


    @abstractmethod
    def get_legal_actions(self):
        '''
        Return a list of legal actions from the current game state
        '''
        pass


    @abstractmethod
    def apply_action(self, action):
        '''
        Implement a function to apply an action to the current state and return the resulting state

        Must return an instance of the child class
        '''
        pass

    
    def action(self, action):

        '''
        Apply an action to the current state, with checks that the move is within the rules.

        Return the new state of the game.
        '''
        
        if self.game_over:
            raise RuntimeError(f"Cannot execute action '{action}': game already over")
        if action not in self.legal_actions:
            raise RuntimeError(f"Cannot execute action '{action}': not a legal action")

        if isinstance(self.move_history, list):
            self.move_history.append(action)

        return self.apply_action(action)
 

    @abstractmethod
    def is_game_over(self):
        '''
        Check if state represents terminated game, return True/False
        '''
        pass


    @abstractmethod
    def check_outcome(self):
        '''
        Return outcome of terminated game
        '''
        pass
    

    def display(self):

        '''
        Show the game. Defaults to just printing the state. 
        '''

        print(f'\nGAME STATE:\n{self.state}\n')
