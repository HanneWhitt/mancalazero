from abc import ABC, abstractmethod


class GameState(ABC):
    
    '''
    Abstract base class for classical games, exposing functions required for AlphaZero algorithm

    Essentially a state tensor with functions attached
    '''

    def __init__(self, state='new_game'):
        if state == 'new_game':
            state = self.new_game_state()
        self.state = state
        self.game_over = self.is_game_over()
        if self.game_over:
            self.legal_actions = None
        else:
            self.legal_actions = self.get_legal_actions()
    

    @abstractmethod
    @staticmethod
    def new_game_state(self):
        '''
        Return the state tensor of a new game
        '''
        pass


    @property
    def state(self):
        print("Getting value...")
        return self._state
    

    @state.setter
    def state(self, state):
        print("Setting value...")
        if self.check_state:
            self.check_state_validity(state)
        self._state = state


    @abstractmethod
    @staticmethod
    def check_state_validity(self, state):
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
        Apply an action to a non-terminated GameState and return the resulting child GameState
        '''
        pass


    def action(self, action):
        
        if self.game_over():
            raise RuntimeError("Cannot execute action '{action}': game already over")
        if action not in self.get_legal_actions():
            raise RuntimeError("Cannot execute action '{action}': not legal")

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
    


if __name__ == '__main__':

    class TestGameState(GameState):
        def apply_action(self, action):
            pass

    tgs = TestGameState(0)

    print(tgs.state)
    
    tgs.state = '1'
    
    print(tgs.state)
