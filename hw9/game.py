import random
import time
import copy

class TeekoPlayer:
    """ An AI player for Teeko. """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its piece color. """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
    
    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return True
    
    def is_drop_phase(self, state):
        """ Determines if the game is in the drop phase. """
        piece_count = sum(row.count(self.my_piece) + row.count(self.opp) for row in state)
        return piece_count < 8

    def make_move(self, state):
        """ Selects a move using minimax with alpha-beta pruning. """
        drop_phase = self.is_drop_phase(state)
        max_depth = 3 if drop_phase else 4
        start_time = time.time()
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        best_move = None

        if drop_phase:
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[r][c] = self.my_piece
                        value = self.min_value(new_state, max_depth - 1, alpha, beta, start_time)
                        if value > best_value:
                            best_value = value
                            best_move = [(r, c)]
                        alpha = max(alpha, best_value)
                        if time.time() - start_time > 4.5:
                            return best_move if best_move else [(random.randint(0, 4), random.randint(0, 4))]
        else:
            my_pieces = [(r, c) for r in range(5) for c in range(5) if state[r][c] == self.my_piece]
            for r, c in my_pieces:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[r][c] = ' '
                        new_state[nr][nc] = self.my_piece
                        value = self.min_value(new_state, max_depth - 1, alpha, beta, start_time)
                        if value > best_value:
                            best_value = value
                            best_move = [(nr, nc), (r, c)]
                        alpha = max(alpha, best_value)
                        if time.time() - start_time > 4.5:
                            return best_move if best_move else [(nr, nc), (r, c)]
        return best_move

    def min_value(self, state, depth, alpha, beta, start_time):
        if time.time() - start_time > 4.5 or depth == 0 or self.game_value(state) != 0:
            return self.heuristic_game_value(state)
        value = float('inf')
        for succ_state in self.succ(state, False):
            value = min(value, self.max_value(succ_state, depth - 1, alpha, beta, start_time))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(self, state, depth, alpha, beta, start_time):
        if time.time() - start_time > 4.5 or depth == 0 or self.game_value(state) != 0:
            return self.heuristic_game_value(state)
        value = -float('inf')
        for succ_state in self.succ(state, True):
            value = max(value, self.min_value(succ_state, depth - 1, alpha, beta, start_time))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def succ(self, state, is_max):
        piece = self.my_piece if is_max else self.opp
        successors = []
        drop_phase = self.is_drop_phase(state)

        if drop_phase:
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[r][c] = piece
                        successors.append(new_state)
        else:
            for r in range(5):
                for c in range(5):
                    if state[r][c] == piece:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                new_state = copy.deepcopy(state)
                                new_state[r][c] = ' '
                                new_state[nr][nc] = piece
                                successors.append(new_state)
        return successors

    def heuristic_game_value(self, state):
        """Enhanced heuristic evaluation using patterns."""
        game_val = self.game_value(state)
        if game_val != 0:
            return game_val  # Return terminal state score
        
        score = 0
        
        # Evaluate patterns for potential winning opportunities
        score += self.evaluate_patterns(state)
        
        # Additional heuristics (optional, e.g., clustering, adjacency)
        for r in range(5):
            for c in range(5):
                if state[r][c] == self.my_piece:
                    score += 1  # Bonus for each piece
                elif state[r][c] == self.opp:
                    score -= 1  # Penalty for opponent's pieces
        
        return score / 100.0  # Normalize score for consistency


    def evaluate_position(self, state, r, c):
        score = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == self.my_piece:
                score += 1
        return score

    def game_value(self, state):
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i] == self.my_piece else -1
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col] == self.my_piece else -1
        return 0

    def opponent_move(self, move):
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def evaluate_patterns(self, state):
        """Evaluate potential winning patterns."""
        score = 0
        
        # Check for three in a row
        for r in range(5):
            for c in range(3):
                window = [state[r][c+i] for i in range(3)]
                score += self.evaluate_window(window)
                
        # Check for three in a column
        for r in range(3):
            for c in range(5):
                window = [state[r+i][c] for i in range(3)]
                score += self.evaluate_window(window)
                
        # Check for three in diagonal
        for r in range(3):
            for c in range(3):
                window = [state[r+i][c+i] for i in range(3)]
                score += self.evaluate_window(window)
                
        return score

    def evaluate_window(self, window):
        """Helper function to evaluate a window of three positions."""
        score = 0
        
        if window.count(self.my_piece) == 2 and window.count(' ') == 1:
            score += 5
        elif window.count(self.opp) == 2 and window.count(' ') == 1:
            score -= 4
            
        return score
    

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
