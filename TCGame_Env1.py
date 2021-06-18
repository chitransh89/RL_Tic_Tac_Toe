from itertools import groupby
from itertools import product
from gym import spaces
import numpy as np
import random


class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def __row_check(self, curr_state):
        for row in range(3):
            # print("Doing Row check")
            # print(3*row, 3*row + 1, 3*row + 2)
            nums = (curr_state[3*row], curr_state[3*row + 1], curr_state[3*row + 2])
            sum_ = sum(nums)
            if sum_ == 15:
                # print(nums, row)
                return True
            sum_ = 0
        return False


    def __col_check(self, curr_state):
        for col in range(3):
            # print("Doing Column Check")
            # print(col, col+3, col+6)
            nums = (curr_state[col], curr_state[col + 3], curr_state[col + 6])
            sum_ = sum(nums)
            if sum_ == 15:
                # print(nums, col)
                return True
            sum_ = 0
        return False


    def __diag_check(self, curr_state):
        # print("Doing Diagonal Check")
        # left diagonal
        num_left_diag = (curr_state[0], curr_state[4], curr_state[8])
        # right diagonal
        num_right_diag = (curr_state[2], curr_state[4], curr_state[6])
        return (sum(num_right_diag) == 15) or (sum(num_left_diag) == 15)


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        # row check
        if self.__row_check(curr_state):
            return True
        
        # col check
        if self.__col_check(curr_state):
            return True

        # diagonal check
        if self.__diag_check(curr_state):
            return True
        
        return False
        
 
    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state):
            return True, 'Win'

        if len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'

        return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 != 0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 == 0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        # Agent Making move
        next_state = self.state_transition(curr_state, curr_action)

        is_game_end, board_state = self.is_terminal(next_state)
        if is_game_end:
            reward = 10 if board_state == 'Win' else 0
        else:
            # Random event move
            _, possible_env_moves = self.action_space(next_state)
            listify_env_moves = list(possible_env_moves)
            random_selector = random.randint(0, (len(listify_env_moves)-1))
            env_move = listify_env_moves[random_selector]
            next_state = self.state_transition(next_state, env_move)

            is_game_end, board_state = self.is_terminal(next_state)
            if is_game_end:
                reward = -10 if board_state == 'Win' else 0
            else:
                reward = -1
        return next_state, reward, is_game_end

    
    def reset(self):
        self.state = [np.nan for _ in range(9)]
        return self.state
