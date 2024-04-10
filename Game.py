from abc import ABC, abstractmethod


class GameState(ABC):
    
    '''
    Abstract base class for classical games, exposing functions required for AlphaZero algorithm

    Essentially a state with functions attached
    '''

    def __init__(self, state=None, check_validity=True):
        if state is None:
            state = self.new_game_state()
        self.check_validity = check_validity
        self.state = state
    

    @staticmethod
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


    def get_observation(self):
        '''
        Return a tensor representing an observation of the state.

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
        Apply an action to the current state and return the resulting state
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
            raise RuntimeError(f"Cannot execute action '{action}': not legal")

        return self.apply_action(action)
 

    @abstractmethod
    def is_game_over(self):
        '''
        Check if state represents terminated game, return True/False
        '''
        pass


    @abstractmethod
    def check_winner(self):
        '''
        Return winner of terminated game
        '''
        pass
    

    def display(self):

        '''
        Show the game. Defaults to just printing the state. 
        '''

        print(f'\nGAME STATE:\n{self.state}\n')



if __name__ == '__main__':

    class CoinFlip(GameState):

        "A stupid game to test the ABC. Flip a coin 10 times, then game over"

        def __init__(self, state=None, check_validity=True):
            super().__init__(state, check_validity)


        def new_game_state(self):
            return [0, 0]
        
        @property
        def coin_side(self):
            return self.state[0]
        
        @property
        def turn_no(self):
            return self.state[1]

        def check_state_validity(self):
            if self.coin_side not in [0, 1]:
                raise ValueError(f'Coin side wrong: {self.coin_side}')

        def get_legal_actions(self):
            return ['flip']
        
        def apply_action(self, action):
            return [1 - self.coin_side, self.turn_no + 1]
        
        def is_game_over(self):
            return self.turn_no > 10

        def check_winner(self):
            return 'me'


    cf = CoinFlip()

    for i in range(12):

        if cf.game_over:
            print('Game over!')
            break

        cf.display()
        new_state = cf.action('flip')
        cf = CoinFlip(new_state)
        