from mancalazero.gamestate import GameState


class CoinFlip(GameState):

    "A stupid game to test the ABC. Flip a coin 10 times, then game over"

    actions_list = ['flip']
    shape = (2, )

    def __init__(self, state=None, move_history=True, check_validity=True):
        super().__init__(state, move_history, check_validity)


    def new_game_state(self):
        return [0, 0]
    
    @property
    def coin_side(self):
        return self.state[0]
    
    @property
    def current_player(self):
        return self.coin_side
    
    @property
    def turn_number(self):
        return self.state[1]

    def check_state_validity(self):
        if self.coin_side not in [0, 1]:
            raise ValueError(f'Coin side wrong: {self.coin_side}')

    def get_legal_actions(self):
        return ['flip']
    
    def apply_action(self, action):
        new_state = [1 - self.coin_side, self.turn_number + 1]
        return CoinFlip(
            new_state,
            self.move_history,
            self.check_validity
        )
    
    def is_game_over(self):
        return self.turn_number > 10

    def check_outcome(self):
        return 'Game over!'


cf = CoinFlip()

for i in range(12):

    if cf.game_over:
        print(cf.check_outcome())
        break

    cf.display()
    print('move history:', cf.move_history)
    cf = cf.action('flip')
    