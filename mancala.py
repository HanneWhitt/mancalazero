import numpy as np


class MancalaBoard:

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

    def __init__(self,
                current_player=1, 
                board_start=None,
                turn_number=0,
                starting_stones=3,
                place_in_opponent_store=False,
                capture_last_stone_in_zero_hole=True,
                no_moves_policy='end_game'):
        
        self.starting_stones = starting_stones

        if board_start is None:
            # New game
            self.board = np.ones(14, dtype=np.uint8)*starting_stones
            self.board[[6, 13]] = 0 # These spaces represent stores, which start empty
        else:
            self.board = np.array(board_start, dtype=np.uint8).reshape(14)

        # Check board validity using the total number of stones - important if board object created from non-start position.
        self.total_stones = starting_stones*12
        self.check_valid()

        # Current player and turn number included for book-keeping but can also be added as features
        self.current_player = current_player 
        self.turn_number = turn_number

        # Variables for rules version
        self.place_in_opponent_store = place_in_opponent_store
        self.capture_last_stone_in_zero_hole = capture_last_stone_in_zero_hole
        assert no_moves_policy in ['end_game', 'pass_back'], "no_moves_policy must be in ['end_game', 'pass_back']"
        self.no_moves_policy = no_moves_policy


    def get_representation(self):
        
        '''
        Add a feature for the current player and the turn number onto the board representation to create a full feature representation of the board. 
        '''

        player_feature = self.current_player - 1
        return np.concatenate([self.board, [player_feature, self.turn_number]])


    def move(self, move):

        '''
        Take in a move representation in index form (0-5 for six possible moves), and apply to board to update:
        - Board representation
        - Current player
        - Turn number
        
        Set everything up such that updated board representation is ready for next move/evaluation
        '''

        legal_moves = self.get_legal_moves()
        assert move in legal_moves, 'Illegal or invalid move!'

        # MOVE STONES
        # Pick up stones
        to_distribute = self.board[move]
        self.board[move] = 0

        # Distribute stones
        position = move
        while to_distribute > 0:
            position = (position + 1) % 14
            if position != 13 or self.place_in_opponent_store:
                to_distribute -= 1
                self.board[position] += 1

        # If final stone was placed in a position on the players side and where there were 0 stones at move beginning,
        # capture stones in hole on opposite side and optionally also the last placed stone depending on variant
        # Stones move to current player store
        if position < 6 and self.board[position] == 1: # If there is only one stone in last position, must have been 0 before this turn.
            opposite_hole = 12 - position
            self.board[6] += self.board[opposite_hole]
            self.board[opposite_hole] = 0
            if self.capture_last_stone_in_zero_hole:
                self.board[6] += 1
                self.board[position] = 0

        # SET UP NEXT BOARD POSITION WHICH IS EVALUATED
        # Play passes to other player unless:
        # i) final stone was placed in players own store
        if position != 6:
            self.change_player()
        
        # ii) The new player (considering (i)) is left with no stones on his side and hence no legal moves
        if self.board[:6].sum() == 0:
            # If pass_back variant, then play goes just goes back to the other side
            if self.no_moves_policy == 'pass_back':
                self.change_player()
            # If end_game variant, the opposite player gains all the stones still in play. This will empty the board and trigger the
            # end_game variant win condition
            else:
                self.board[13] += self.board[7:13].sum()
                self.board[7:13] = 0
                
        self.turn_number += 1


    # This function changes both the current player value and rotates the board view. This is so that moves from either player can be
    # evaluated equivalently by network
    def change_player(self):
        self.current_player = 3 - self.current_player
        self.board = np.concatenate([self.board[7:], self.board[:7]])


    def get_legal_moves(self, mask_format=False):
        if mask_format:
            raise NotImplementedError
        else:
            legal_move_indices = np.nonzero(self.board[:6])[0]
            return legal_move_indices


    def check_score(self):
        
        # Which position is store for which player depends on current player
        if self.current_player == 1:
            player_1_score = self.board[6]
            player_2_score = self.board[-1]
        else:
            player_1_score = self.board[-1]
            player_2_score = self.board[6]

        return player_1_score, player_2_score


    def game_over(self):

        '''
        Check if game is over

        In end_game mode, move() will move last stones into correct store; hence the termination condition for both end_game and pass_back is just that all non-store spaces are 0
        '''

        return self.board[:6].sum() == self.board[7:13].sum() == 0


    def check_winner(self, p1_victory_value = 1, p2_victory_value = 2, draw_value = None):

        '''
        Return winner of terminated game.
        '''

        player_1_score, player_2_score = self.check_score()

        if player_2_score > player_1_score:
            return p2_victory_value
        elif player_1_score > player_2_score:
            return p1_victory_value
        else:
            return draw_value


    def check_valid(self):

        '''
        Apply a validity check to the board
        '''

        assert self.board.sum() == self.total_stones, 'Wrong number of stones on board!'
        

    def display(self):
        current_player_store = self.board[6]
        opposite_player_store = self.board[-1]
        lower_row = self.board[:6]
        upper_row = np.flip(self.board[7:-1])
        print('\n\n\n PLAYER', 3 - self.current_player, '  ')
        print(' ', upper_row, '  ')
        print(opposite_player_store, '             ', current_player_store)
        print(' ', lower_row, '  ')
        print(' PLAYER', self.current_player, '  ')
        print("\nPlayer {}'s turn.".format(self.current_player))


    def play_in_console(self, show_features = False):
        player_entry = ''
        while player_entry != 'exit':
            self.display()
            if self.game_over():
                print('\nGame Over!')
                print(self.check_winner(p1_victory_value = 'Player 1 wins!', p2_victory_value = 'Player 2 wins!', draw_value = 'Draw!'))
                break
            else:
                if show_features:
                    print(f'\nFeatures: {self.get_board_state()}\n')
                player_entry = input("Enter a move 1-6 or 'exit': ")
                if player_entry in ['1', '2', '3', '4', '5', '6']:
                    self.move(int(player_entry) - 1)


    def __copy__(self):
        return MancalaBoard(current_player=self.current_player, 
                            board_start=self.board,
                            turn_number=self.turn_number,
                            starting_stones=self.starting_stones,
                            place_in_opponent_store=self.place_in_opponent_store,
                            capture_last_stone_in_zero_hole=self.capture_last_stone_in_zero_hole,
                            no_moves_policy=self.no_moves_policy)
        
        

if __name__ == '__main__':

    # rep = None #[0, 0, 0, 0, 0, 5, 31, 0, 0, 0, 0, 0, 0, 0]
    # current_player = 1

    board1 = MancalaBoard()
    board1.move(1)
    board2 = board1.copy()
    board2.move(2)

    board1.display()
    board2.display()


