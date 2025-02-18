import numpy as np
import random as rand
import reversi
import time

class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num

    def make_move(self, state):
        start_time = time.time()
        num_zeros = np.count_nonzero(state.board == 0)
        
        if self.move_num < 4:
            duration = 0.05
        elif (num_zeros < 24 and self.move_num < 20):
            duration = 8.0
        elif (num_zeros < 44 and self.move_num < 40):
            duration = 12.0
        else:
            duration = 5.0

        valid_moves = state.get_valid_moves()
        best_move = valid_moves[0] 
        best_score = float('-inf')
        current_depth = 1
        

        while True:

            if time.time() - start_time >= duration:
                break
            
            depth_best_move = None
            depth_best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            

            for move in valid_moves:
                if time.time() - start_time >= duration:
                    break
                
                temp_state = self.make_test_move(state, move)
                score = self.minimax(temp_state, current_depth, False, alpha, beta)
                
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                
                alpha = max(alpha, depth_best_score)
            

            if depth_best_move is not None:
                best_move = depth_best_move
                best_score = depth_best_score
            

            current_depth += 1
            

            time.sleep(0.001)
        
        return best_move

    def heuristic_evaluation(self, state, is_maximizing):

        WEIGHTS = np.array([
            [100, -20,  10,   5,   5,  10, -20, 100],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [ 10,  -2,   1,   1,   1,   1,  -2,  10],
            [  5,  -2,   1,   0,   0,   1,  -2,   5],
            [  5,  -2,   1,   0,   0,   1,  -2,   5],
            [ 10,  -2,   1,   1,   1,   1,  -2,  10],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [100, -20,  10,   5,   5,  10, -20, 100]
        ])
        
        player = 1 if is_maximizing else 2
        opponent = 2 if is_maximizing else 1
        
        player_pos = np.where(state.board == player, 1, 0)
        opponent_pos = np.where(state.board == opponent, 1, 0)
        position_score = np.sum(player_pos * WEIGHTS) - np.sum(opponent_pos * WEIGHTS)
        

        mobility = len(state.get_valid_moves())
        
        corners = [(0,0), (0,7), (7,0), (7,7)]
        player_corners = sum(1 for x,y in corners if state.board[x][y] == player)
        opponent_corners = sum(1 for x,y in corners if state.board[x][y] == opponent)
        corner_score = (player_corners - opponent_corners) * 25
        

        piece_diff = (np.count_nonzero(state.board == player) - 
                     np.count_nonzero(state.board == opponent))
        
        total_score = (position_score * 1.0 + 
                      mobility * 2.0 + 
                      corner_score * 5.0 + 
                      piece_diff * 0.1)
        
        return total_score

    def minimax(self, state, depth, is_maximizing, alpha, beta):
        if depth == 0 or not state.get_valid_moves() or np.all(state.board):
            return self.heuristic_evaluation(state, is_maximizing)
        
        valid_moves = state.get_valid_moves()
        
        if is_maximizing:
            best_score = float('-inf')
            for move in valid_moves:
                new_state = self.make_test_move(state, move)
                score = self.minimax(new_state, depth - 1, False, alpha, beta)
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in valid_moves:
                new_state = self.make_test_move(state, move)
                score = self.minimax(new_state, depth - 1, True, alpha, beta)
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def make_test_move(self, state, move):
        new_state = reversi.ReversiGameState(np.copy(state.board), state.turn)
        row, col = move
        new_state.board[row][col] = state.turn
        
        new_state.turn = 1 if state.turn == 2 else 2
        
        return new_state
