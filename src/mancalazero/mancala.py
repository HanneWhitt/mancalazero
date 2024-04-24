import numpy as np
from mancalazero.Game import GameState


class MancalaBoard(GameState):

    '''
    An object to represent a game of mancala, implementing board representations, legal moves, game termination/win conditions etc.

    Also facility for playing a game in console for debugging/fun :)

    Includes a few options on rules:

    1) How many starting stones in each hole (3 or 4 both popular)
    2) Whether player should place stones in opponent's store on way round (both variants exist)
    3) Whether, when a player places his last stone in an empty hole on his side, he captures just the stones in the hole on the opposite side, or whether he captures his own last-placed stone as well
    4) Whether, when a player cannot make a move (no stones on his side):
        i) The game ends, and the other player gains all the remaining stones in play (no_moves_policy='end_game')
        ii) Play simply passes back to other player ('pass_back')

    '''

    def __init__(
        self,
        state=None,
        move_history=True,
        check_validity=True,
        starting_stones=3,
        place_in_opponent_store=False,
        capture_last_stone_in_zero_hole=True,
        no_moves_policy='end_game'
    ):

        # Variables for rules version
        self.starting_stones = starting_stones
        self.place_in_opponent_store = place_in_opponent_store
        self.capture_last_stone_in_zero_hole = capture_last_stone_in_zero_hole
        if no_moves_policy not in ['end_game', 'pass_back']:
            raise ValueError("no_moves_policy must be in ['end_game', 'pass_back']")
        self.no_moves_policy = no_moves_policy

        # Initialise parent
        super().__init__(state, move_history, check_validity)



    def new_game_state(self):
        '''
        Return the state of a new game
        '''
        start_state = np.ones(16, dtype=np.uint8)*self.starting_stones
        # These spaces represent stores, which start empty, current player indicator, and turn number
        start_state[[6, 13, -2, -1]] = 0
        return start_state
    

    @property
    def board(self):
        '''
        Return the part of the state vector representing the mancala board
        '''
        return self.state[:14]


    @property
    def current_player(self):
        '''
        Return the part of the state vector representing the current player (0 or 1)
        '''
        return self.state[-2]


    @property
    def turn_number(self):
        '''
        Return the part of the state vector representing the turn number
        '''
        return self.state[-1]


    def check_state_validity(self):
        # Check total number of stones on board
        if not self.board.sum() == 12*self.starting_stones:
            raise RuntimeError('Wrong number of stones on board!')
        # Check player index
        if self.current_player not in [0, 1]:
            raise RuntimeError('Player index not 0 or 1')
        # Check all elements of state tensor positive
        if not np.all(self.state >= 0):
            raise RuntimeError('Negative value in state tensor')
        #print('New state validity OK')


    def get_observation(self):
        '''
        Return the state of the game as seen by the current player
        '''
        observation = self.state.copy()
        if self.current_player == 1:
            observation = self.reverse_view(observation)
        return observation
    

    def apply_action(self, action):

        '''
        Take in a move representation in index form (0-5 for six possible moves), and apply to board to update:
        - Board representation
        - Current player
        - Turn number
        
        Set everything up such that updated board representation is ready for next move/evaluation
        ''' 

        # Useful shorthand to rotate the board if appropriate
        new_state = self.get_observation()

        # MOVE STONES
        # Pick up stones
        to_distribute = new_state[action]
        new_state[action] = 0

        # Distribute stones
        position = action
        while to_distribute > 0:
            position = (position + 1) % 14
            if position != 13 or self.place_in_opponent_store:
                to_distribute -= 1
                new_state[position] += 1

        # If final stone was placed in a position on the players side and where there were 0 stones at move beginning,
        # capture stones in hole on opposite side and optionally also the last placed stone depending on variant
        # Stones move to current player store
        if position < 6 and new_state[position] == 1: # If there is only one stone in last position, must have been 0 before this turn.
            opposite_hole = 12 - position
            new_state[6] += new_state[opposite_hole]
            new_state[opposite_hole] = 0
            if self.capture_last_stone_in_zero_hole:
                new_state[6] += 1
                new_state[position] = 0

        # Play now passes to other player unless:
        # i) final stone was placed in players own store, in which case they get another go
        # ii) We are playing the pass-back variant, and the other player has no stones left to play
        current_player_has_stones = new_state[:6].sum() != 0
        other_player_has_stones = new_state[7:-3].sum() != 0
        ended_in_current_player_store = position == 6
        
        if self.no_moves_policy == 'end_game':
            if not ended_in_current_player_store:
                new_state[-2] = 1 - self.current_player
        
        elif self.no_moves_policy == 'pass_back':
            if not current_player_has_stones:
                new_state[-2] = 1 - self.current_player
            else:
                if other_player_has_stones and not ended_in_current_player_store:
                    new_state[-2] = 1 - self.current_player



        # Rotate the board back again if appropriate
        if self.current_player == 1:
            new_state = self.reverse_view(new_state)

        # Increment turn number
        new_state[-1] = self.turn_number + 1

        return MancalaBoard(
            new_state,
            self.move_history,
            self.check_validity,
            self.starting_stones,
            self.place_in_opponent_store,
            self.capture_last_stone_in_zero_hole,
            self.no_moves_policy
        )


    # This function rotates the board view to see game from other player's perspective
    @staticmethod
    def reverse_view(state):
        return np.concatenate([state[7:-2], state[:7], state[-2:]])


    def get_legal_actions(self):
        # Shorthand to view board from current player's perspective
        observation = self.get_observation()
        return np.nonzero(observation[:6])[0]


    def check_score(self):
        
        if self.game_over:
            player_0_score = self.board[:7].sum()
            player_1_score = self.board[7:].sum()
        else:
            player_0_score = self.board[6]
            player_1_score = self.board[-1]

        return player_0_score, player_1_score


    def is_game_over(self):
        '''
        Check if game is over

        Differs depending on rule for when one side is empty
        '''
        observation = self.get_observation()
        current_player_no_stones = observation[:6].sum() == 0

        if current_player_no_stones:
            if self.no_moves_policy == 'end_game':
                # No stones on current player side ends game
                return True
            elif self.no_moves_policy == 'pass_back':
                other_player_no_stones = observation[7:-3].sum() == 0
                # No stones on both sides ends game
                return other_player_no_stones
        return False


    def check_outcome(self):

        '''
        Return winner of terminated game.
        '''

        if not self.game_over:
            raise RuntimeError('Asked to check winner of non-terminated game')

        player_0_score, player_1_score = self.check_score()

        if player_0_score > player_1_score:
            return 1
        elif player_1_score > player_0_score:
            return -1
        else:
            return 0
        

    def display(self):
        player_0_store = self.board[6]
        player_1_store = self.board[-1]
        lower_row = self.board[:6]
        upper_row = np.flip(self.board[7:-1])
        display_string = f"""
{upper_row}
{player_1_store}             {player_0_store}
{lower_row}"""
        return display_string


    def play_in_console(self):
        player_entry = ''
        while player_entry != 'exit':
            self.display()
            if self.game_over:
                print('\nGame Over!')
                print('Game value:', self.check_winner())
                break
            else:
                player_entry = input("Enter a move 1-6 or 'exit': ")
                if player_entry in ['1', '2', '3', '4', '5', '6']:
                    self.state = self.action(int(player_entry) - 1).state

    
