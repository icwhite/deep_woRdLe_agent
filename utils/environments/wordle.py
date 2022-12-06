import numpy as np
from english_words import english_words_set
import gym
from gym import spaces
from utils.exploration.base_exploration_model import BaseExplorationModel
from gym.utils.env_checker import check_env
from collections import deque
from utils.infrastructure.logger import Logger


from collections import defaultdict
import re

class Wordle(gym.Env): 
    
    def __init__(self, 
                 n_boards: int = 1,
                 n_letters: int = 5,
                 n_guesses: int = 6,
                 answers: list = None,
                 subset_valid_words: int = 0,
                 subset_answers: int = 0,
                 seed: int = 0,
                 keep_answers_on_reset: bool = False,
                 exploration_model: BaseExplorationModel = BaseExplorationModel(),
                 valid_words: list = None,
                 logdir: str = None,
                 logging_freq: int = 500,
                 reward: str = "elimination"):
        
        """
        n_boards: number of boards that are played at once
        n_letters: number of letters per word
        n_guesses: number of guesses per board
        answers: list of answers. If None then answers are selected at random
        seed: seed for selecting random answers. If None then different answers will be selected each time
        keep_answers_on_reset: whether we should select new answers on each reset.
        exploration_model: the exploration model to be used
        logdir: the logging directory
        logging_freq: how often we log the win ratio
        reward: chooses which reward function we want to use
        """
        
        
        # Store attributes 
        self.n_boards = n_boards
        self.n_letters = n_letters
        self.n_guesses = n_guesses
        self.seed = seed 
        np.random.seed(self.seed)
        self.keep_answers_on_reset = keep_answers_on_reset
        self.exploration_model = exploration_model
        
        # Create the list of valid words of length n_letters
        self.valid_words = valid_words if valid_words is not None else [word.lower() for word in english_words_set if len(word) == self.n_letters]
        self.valid_words = [word for word in self.valid_words if "'" not in word and "." not in word and "&" not in word]
        self.valid_words = sorted(self.valid_words)

        # pick which reward function we want to use
        if reward == "elimination":
            self.reward_function = self.compute_single_board_sparse_reward
        elif reward == "info":
            self.reward_function = self.compute_single_board_info_gain_reward
        elif reward == "sparse":
            self.reward_function = self.compute_single_board_sparse_reward
        else:
            raise NotImplementedError

        # self.explore_weight_schedule = explore_weight_schedule
        # self.exploit_weight_schedule = exploit_weight_schedule

        if subset_valid_words:
            self.valid_words = np.random.choice(self.valid_words, subset_valid_words).tolist()
        if subset_answers:
            self.valid_answers = np.random.choice(self.valid_words, subset_answers).tolist()
        else:
            self.valid_answers = self.valid_words

        
        # Create answers. If we pass a list it will set as answers. Otherwise it will generate a list 
        # of size n_boards of random answers. 
        # Alternatively, we can pass a seed for reproducibility
        if answers is not None: 
            assert(isinstance(answers, list))
            assert(len(answers) == self.n_boards)
            assert([len(answer) == self.n_letters for answer in answers])
            self.answers = answers
        else: 
            self.answers = np.random.choice(self.valid_answers, self.n_boards).tolist()

        # create encoder and decoder for later use in self._encode and self._decode
        self.encoder = dict(zip(list("abcdefghijklmnopqrstuvwxyz"), np.arange(26)))
        self.decoder = dict(zip(np.arange(26), list("abcdefghijklmnopqrstuvwxyz")))

        self.encoded_answers = [self._encode(answer) for answer in self.answers]
            
            
        # Action Space 
        self.action_space = spaces.Discrete(len(self.valid_words))
        
        # Observation Space 
        self.obs_dims = self.n_boards * self.n_letters * self.n_guesses * 2 # 2 for letters/colors
        self.observation_space = spaces.Box(low = -1,
                                            high = 25, 
                                            shape = (self.obs_dims,), 
                                            dtype = int)
        # Initialize State
        self.state = -1 * np.ones(self.obs_dims, dtype = int)
        
        # Initialize tracking variables
        self.guess_count = 0
        self.wins = [False] * self.n_boards # tracks which boards have been won already
        self.done = False # overall is the game done
        self.board_guess_counts = [self.guess_count] * self.n_boards # guess count for each board
        self.green_letters = [] # letters that we know are green
        self.yellow_letters = [] # letters that are yellow (runs into case of guessing the same letter in two spots and it's yellow both times)
        # but we'll ignore for now

        # logging
        self.logging_freq = logging_freq
        self.victory_buffer = deque(maxlen= self.logging_freq)
        self.logger = Logger(logdir)
        self.num_games = 0

        self.possible_words = self.valid_words
        self.alphabet = ['a', 'b', 'c' , 'd', 'e', 'f', 'g', 'h', 'i', 
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z']
        
    def _encode(self, word: str): 
        """
        Inputs a word of length n_letters as a string and maps to a list of ints corresponding to 
        each letter a = 0, b = 1, ..., z = 25
        """
        return [self.encoder[letter] for letter in word]    
    
    def _decode(self, letters_list: list):
        """
        :param letters_list: list of letters where each value is from 0 to 25 mapping to a-z
        :return: a word/string containing the corresponding letters
        """
        return [self.decoder[idx] for idx in letters_list]               
        
    def _convert_state_to_grids(self, state_1d): 
        
        """
        Converts the 1d array into a list of dictionaries. Note that the 2
        comes from the fact that we have separate grids for letters and colors.
        [ {'letters': [...], 'colors': [...]}, {'letters': [...], 'colors': [...]}, ..., ]
        """
        
        # Convert the 1d array into an array of size n_boards x 2 * n_guesses x n_letters
        state = state_1d.reshape(( self.n_boards, 2 * self.n_guesses, self.n_letters, ))
        
        # Convert into list of dictionaries 
        state_list = [{'letters': board[:self.n_guesses], 'colors': board[self.n_guesses:]} for board in state]
        
        return state_list
    
    def _convert_grids_to_state(self, boards): 
        
        """
        Converts the states from list of dictionaries to a single flattened list of size 
        n_boards * n_guesses * n_letters * 2 (2 because we have separate grids for letters and colors)
        
        boards: list of dict of form [ {'letters': [...], 'colors': [...]}, {'letters': [...], 'colors': [...]}, ..., ]
        """
        
        return np.array([np.append(board['letters'], board['colors']) for board in boards], dtype=int).flatten()

    def compute_single_board_sparse_reward(self, board: dict, board_guess_count: int,\
                                           decoded_answer:str, decoded_guess: str):


        """
        Because green is 2, the max score is 2 x number of letters
        We give a score of -1 for any bad guessses

        board: dict of form {'letters': [...], 'colors': [...]}
        """


        score = np.sum(board['colors'][board_guess_count])
        max_score = 2 * self.n_letters
        reward = 1 if score == max_score else -1

        # return reward
        return reward

    def compute_single_board_info_gain_reward(self, board: dict, board_guess_count: int, decoded_answer, decoded_guess):

        """
        Because green is 2, the max score is 2 x number of letters
        We give a score of -1 for any bad guessses

        board: dict of form {'letters': [...], 'colors': [...]}
        """

        new_greens, new_yellows, other_letters = 0, 0, 0
        for guess_letter, answer_letter in zip(decoded_guess, decoded_answer):

            if (guess_letter == answer_letter) and (guess_letter not in self.green_letters):
                new_greens += 1
                self.green_letters.append(guess_letter)
            elif (guess_letter in decoded_answer) and (guess_letter not in self.yellow_letters):
                new_yellows += 1
                self.yellow_letters.append(guess_letter)
            else:
                other_letters += 1

        score = 2 * new_greens + 1 * new_yellows - other_letters

        # Check if won board
        won = (decoded_guess == decoded_answer)
        if not won:
            score -= 10

        return score

    def compute_single_board_reward(self, board: dict, board_guess_count: int, decoded_answer: str,
                                    decoded_guess: str): 
        
        """
        Computes the number of words eliminated
        """
        
        # Green dictionary with keys = 0, 1, ..., n_letter and value = '' if no green or letter if green
        green_letters = dict(zip(range(self.n_letters), ['']*self.n_letters))
        
        # Yellow dict with keys = 0, 1, ..., n_letters and values are lsits of yellows
        yellow_letters = defaultdict(list)
        
        # list of gray letters
        gray_letters = []
        for idx, (guess_letter, answer_letter) in enumerate(zip(decoded_guess, decoded_answer)): 

            # Green Letter
            if guess_letter == answer_letter: 
                green_letters.update({idx: guess_letter})

            # yellow letter
            elif guess_letter in decoded_answer: 
                yellow_letters[idx].append(guess_letter)

            # Gray letter
            else: 
                gray_letters.append(guess_letter)
                
                
        # Remove gray letters from the alphabet
        self.alphabet = sorted(set(self.alphabet) - set(gray_letters))
        
        # Create the new pattern
        pattern = r''
        for i in range(self.n_letters): 

            # Check if there is green or yellow
            is_green = green_letters[i] != ''
            has_yellow = len(yellow_letters[i])  > 0

            if is_green: 
                # if green then it should just be that letter as the only option
                letter_pattern = '[' + green_letters[i] + ']'

            elif has_yellow: 
                
                # if yellow then it's the alphabet minus the letters that can't be there
                letter_alphabet = [letter for letter in self.alphabet if letter not in yellow_letters[i]]
                letter_pattern = '[' + ''.join(letter_alphabet) + ']'

            else: 
                # otherwise just the remaining alphabet
                letter_pattern = '[' + ''.join(self.alphabet) + ']'

            pattern += letter_pattern
            
        # Filter possible words 
        new_possible_words = [word for word in self.possible_words if bool(re.match(pattern, word))]


        # Compute reward 
        reward = (len(self.possible_words) - len(new_possible_words))/len(self.possible_words) - 1
        # print(f'Reduced words from {len(self.possible_words)} to {len(new_possible_words)}')
        self.possible_words = new_possible_words
            
        # Check if the board won
        if decoded_guess == decoded_answer or (len(new_possible_words) == 1 and board_guess_count < 6):
            reward = 1
            
        return reward
   
    def update_single_board(self, 
                            board: dict, 
                            action: int, 
                            encoded_answer: list,
                            board_guess_count: int,
                            board_win: bool): 
        
        """
        Takes the action as index from list of actions and converts to words/list of letter-numbers
        and then updates the board
        {'green': 2, 
         'yellow': 1, 
         'gray': 0, 
         'empty': -1}
         
         board: dict of form {'letters': [...], 'colors': [...]}
         action: integer for the index of self.valid_words selected as the guess
         encoded_answer: the board answer encoded into list of integers
         board_win: boolean for whether the board has already been won or not. If so, it doesn't update anythign 
         (i.e. doesn't make the guess)
         
        """
        
        # If the game is over, we can't make any guesses
        if self.done: 
            return board, 0, True, board_guess_count
        
        # If the board is complete, we can't make any guesses
        elif board_win: 
            
            return board, 0, True, board_guess_count
        
        else: 
                    
            # Grab action
            decoded_action = self.valid_words[action]
            encoded_action = self._encode(decoded_action)

            # Update letters grid 
            board['letters'][board_guess_count] = encoded_action

            # Insert new color records
            board['colors'][board_guess_count] = self._new_colors(encoded_answer, encoded_action)


            # Compute board reward 
            decoded_answer = self._decode(encoded_answer)

            board_reward = self.reward_function(board, board_guess_count, decoded_answer, decoded_action)

            board_win = len(self.possible_words) == 1 or decoded_answer == decoded_action

            # Increment guess count on that board
            board_guess_count += 1
            
            return board, board_reward, board_win, board_guess_count

    def _new_colors(self, encoded_answer, encoded_action):
        """
        Function for finding the colors in the next board state.
        Returns a list of [0 0 2 1 0] where 2 refers to a green letter,
        1 refers to a yellow letter and 0 refers to a gray letter.
        """
        # Update colors
        new_colors = []
        for answer_letters, guess_letter in zip(encoded_answer, encoded_action):

            # Green letter (i.e. correct letter in correct spot)
            if guess_letter == answer_letters:
                new_colors.append(2)
                if guess_letter not in self.green_letters:
                    self.green_letters.append(guess_letter)

            # Yellow letter (i.e. correct letter in incorrect spot)
            elif guess_letter in encoded_answer:
                new_colors.append(1)
                if guess_letter not in self.green_letters:
                    self.yellow_letters.append(guess_letter)

            # Gray letter (i.e. incorrect letter)
            else:
                new_colors.append(0)

        return new_colors
    
    def step(self, action: int): 
        
        """
        Updates each board using the same action and then checks rewards/increments state/etc
        
        action: integer for the index of self.valid_words selected as the guess
        """
        
        # Convert the 1d state into grids for easier use 
        state_list = self._convert_state_to_grids(self.state)
        
        # Initialize stats for updating all boards to track everything
        # these will update the global attributes later
        step_boards = []
        step_rewards = []
        step_wins = []
        step_board_guess_counts = []
        for idx, board in enumerate(state_list): 

            # Get board answer and whether we've finished this board
            encoded_answer = self.encoded_answers[idx]
            board_win = self.wins[idx]
            board_guess_count = self.board_guess_counts[idx]


            # Update each board 
            new_board, board_reward, win, board_guess_count = self.update_single_board(board, 
                                                                                       action, 
                                                                                       encoded_answer, 
                                                                                       board_guess_count,
                                                                                       board_win)


            # Append step information 
            step_boards.append(new_board)
            step_rewards.append(board_reward)
            step_wins.append(win)
            step_board_guess_counts.append(board_guess_count)


        # Compute total reward as mean of reward across n_boards. Picked mean so it's same reward across envs
        reward = np.mean(step_rewards)

        # Increment guess count 
        self.guess_count += 1


        # Check stopping conditions
        win = np.all(step_wins)
        reached_max_guesses = self.guess_count == self.n_guesses
        self.done = bool((reached_max_guesses) or (win))

        # Update info
        self.wins = step_wins
        self.board_guess_counts = step_board_guess_counts
        self.info = {'guess_count': self.guess_count, 
                     'boards_win': self.wins, 
                     'board_guess_counts': self.board_guess_counts, 
                     'win': win} 

        # Convert grids back to 1d state
        self.state = self._convert_grids_to_state(step_boards)

        exploration_bonus = self.compute_bonus(step_boards, action, step_board_guess_counts)

        # change this so that they have a schedule

        reward = self.explore_weight * exploration_bonus + self.exploit_weight * reward

        return self.state, reward, self.done, self.info

    def compute_bonus(self, state, action):
        """
        :param state: the list of dictionaries containing the board state
        :param action: the string of the most recent action
        :return: the exploration bonus corresponding to this particular state and action
        """
        decoded_action = self.valid_words[action]
        return self.exploration_model.compute_bonus(state, decoded_action)
    
    def reset(self, seed = None, return_info = False):
        
        """
        Resets the environment
        
        seed: random seed for selecting random answers. If none, the random answers won't be reproducible
        return_info: boolean for whether we should return the info from the last game
        """

        # log whether we won or not to victory buffer
        self.victory_buffer.append(np.all(self.wins))
        self.num_games += 1
        if not self.num_games % self.logging_freq:
            logs = {
                "win ratio": self._win_ratio()
            }
            self.do_logging(logs, self.num_games)
    
        # Create answers. We can keep answers as well.
        if self.keep_answers_on_reset:
            self.answers = self.answers
            self.encoded_answers = [self._encode(answer) for answer in self.answers]
        else: 
            self.answers = np.random.choice(self.valid_answers, self.n_boards).tolist()
            self.encoded_answers = [self._encode(answer) for answer in self.answers]
                        
        # Initialize State
        self.state = -1 * np.ones(self.obs_dims, dtype = int)
        
        # Initialize tracking variables
        self.guess_count = 0
        self.wins = [False] * self.n_boards # tracks which boards have been won already
        self.done = False # overall is the game done
        self.board_guess_counts = [self.guess_count] * self.n_boards # guess count for each board
        
        self.green_letters = [] # letters that we know are green
        self.yellow_letters = [] # letters that are yellow (runs into case of guessing the same letter in two spots and it's yellow both times)
        # but we'll ignore for now
        
        self.possible_words = self.valid_words
        self.alphabet = ['a', 'b', 'c' , 'd', 'e', 'f', 'g', 'h', 'i', 
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z']
    
        if return_info:
            return self.state, self.info
                        
        return self.state

    def _win_ratio(self):
        """
        Computes the win ration of games currently in the victory buffer.
        :return: the win ratio
        """
        wins = sum(self.victory_buffer)
        return wins/len(self.victory_buffer)

    def do_logging(self, logs, num_games):
        """
        :param logs: dictionary containing values to be logged
        :param num_games: the number of games
        :return: logs values to tensorboard
        """
        print(f"Number of Games: {num_games}")
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, num_games)
        print("\n")

# class Wordle(gym.Env): 
    
    # def __init__(self,
    #              n_boards: int = 1,
    #              n_letters: int = 5,
    #              n_guesses: int = 6,
    #              answers: list = None,
    #              subset_valid_words: int = 0,
    #              subset_answers: int = 0,
    #              seed: int = None,
    #              keep_answers_on_reset: bool = False):
    #
    #     """
    #     n_boards: number of boards that are played at once
    #     n_letters: number of letters per word
    #     n_guesses: number of guesses per board
    #     answers: list of answers. If None then answers are selected at random
    #     seed: seed for selecting random answers. If None then different answers will be selected each time
    #     keep_answers_on_reset: whether we should select new answers on each reset.
    #     """
    #
    #
    #     # Store attributes
    #     self.n_boards = n_boards
    #     self.n_letters = n_letters
    #     self.n_guesses = n_guesses
    #     self.seed = seed
    #     self.keep_answers_on_reset = keep_answers_on_reset
    #
    #     # Create the list of valid words of length n_letters
    #     self.valid_words = [word.lower() for word in english_words_set if len(word) == self.n_letters]
    #     self.valid_words = [word for word in self.valid_words if "'" not in word and "." not in word and "&" not in word]
    #     self.valid_words = sorted(self.valid_words)
    #
    #     if subset_valid_words:
    #         self.valid_words = np.random.choice(self.valid_words, subset_valid_words).tolist()
    #     if subset_answers:
    #         self.valid_answers = np.random.choice(self.valid_words, subset_answers).tolist()
    #     else:
    #         self.valid_answers = self.valid_words
    #
    #     # Create answers. If we pass a list it will set as answers. Otherwise it will generate a list
    #     # of size n_boards of random answers.
    #     # Alternatively, we can pass a seed for reproducibility
    #     if answers is not None:
    #         assert(isinstance(answers, list))
    #         assert(len(answers) == self.n_boards)
    #         assert([len(answer) == self.n_letters for answer in answers])
    #         self.answers = answers
    #     else:
    #         self.answers = np.random.choice(self.valid_words, self.n_boards).tolist()
    #
    #     self.encoded_answers = [self._encode(answer) for answer in self.answers]
    #
    #
    #     # Action Space
    #     self.action_space = spaces.Discrete(len(self.valid_words))
    #
    #     # Observation Space
    #     self.obs_dims = self.n_boards * self.n_letters * self.n_guesses * 2 # 2 for letters/colors
    #     self.observation_space = spaces.Box(low = -1,
    #                                         high = 25,
    #                                         shape = (self.obs_dims,),
    #                                         dtype = int)
    #     # Initialize State
    #     self.state = -1 * np.ones(self.obs_dims, dtype = int)
    #
    #     # Initialize tracking variables
    #     self.guess_count = 0
    #     self.wins = [False] * self.n_boards # tracks which boards have been won already
    #     self.done = False # overall is the game done
    #     self.board_guess_counts = [self.guess_count] * self.n_boards # guess count for each board
    #     self.green_letters = [] # letters that we know are green
    #     self.yellow_letters = [] # letters that are yellow (runs into case of guessing the same letter in two spots and it's yellow both times)
    #     # but we'll ignore for now
    #
    # def _encode(self, word: str):
    #     """
    #     Inputs a word of length n_letters as a string and maps to a list of ints corresponding to
    #     each letter a = 0, b = 1, ..., z = 25
    #     """
    #     self.encoder = dict(zip(list("abcdefghijklmnopqrstuvwxyz"), np.arange(26)))
    #     return [self.encoder[letter] for letter in word]
    #
    # def _convert_state_to_grids(self, state_1d):
    #
    #     """
    #     Converts the 1d array into a list of dictionaries. Note that the 2
    #     comes from the fact that we have separate grids for letters and colors.
    #     [ {'letters': [...], 'colors': [...]}, {'letters': [...], 'colors': [...]}, ..., ]
    #     """
    #
    #     # Convert the 1d array into an array of size n_boards x 2 * n_guesses x n_letters
    #     state = state_1d.reshape(( self.n_boards, 2 * self.n_guesses, self.n_letters, ))
    #
    #     # Convert into list of dictionaries
    #     state_list = [{'letters': board[:self.n_guesses], 'colors': board[self.n_guesses:]} for board in state]
    #
    #     return state_list
    #
    # def _convert_grids_to_state(self, boards):
    #
    #     """
    #     Converts the states from list of dictionaries to a single flattened list of size
    #     n_boards * n_guesses * n_letters * 2 (2 because we have separate grids for letters and colors)
    #
    #     boards: list of dict of form [ {'letters': [...], 'colors': [...]}, {'letters': [...], 'colors': [...]}, ..., ]
    #     """
    #
    #     return np.array([np.append(board['letters'], board['colors']) for board in boards], dtype=int).flatten()
    #
    # def compute_single_board_reward(self, board: dict, board_guess_count: int):
    #
    #     """
    #     Because green is 2, the max score is 2 x number of letters
    #     We give a score of -1 for any bad guessses
    #
    #     board: dict of form {'letters': [...], 'colors': [...]}
    #     """
    #
    #
    #     # score = np.sum(board['colors'][board_guess_count])
    #     # max_score = 2 * self.n_letters
    #     # reward = 1 if score == max_score else -1
    #
    #     # # return reward
    #     # return reward, score == max_score
    #
    #
    #     # Grab the letters and colors
    #     guess_letters, guess_colors = board['letters'][board_guess_count], board['colors'][board_guess_count]
    #
    #     # Compute information gain
    #     score = 0
    #     for letter, color in zip(guess_letters, guess_colors):
    #         # green scores
    #         if (color == 2) and (letter not in self.green_letters):
    #             score += 2
    #         elif (color == 1) and (letter not in self.yellow_letters):
    #             score += 1
    #         else:
    #             score -= 1
    #
    #     # Check if won board
    #     won = np.sum(guess_colors) == 2 * self.n_letters
    #     if not won:
    #         score -= 10
    #
    #     return score, won
    #
    # def update_single_board(self,
    #                         board: dict,
    #                         action: int,
    #                         encoded_answer: list,
    #                         board_guess_count: int,
    #                         board_win: bool):
    #
    #     """
    #     Takes the action as index from list of actions and converts to words/list of letter-numbers
    #     and then updates the board
    #     {'green': 2,
    #      'yellow': 1,
    #      'gray': 0,
    #      'empty': -1}
    #
    #      board: dict of form {'letters': [...], 'colors': [...]}
    #      action: integer for the index of self.valid_words selected as the guess
    #      encoded_answer: the board answer encoded into list of integers
    #      board_win: boolean for whether the board has already been won or not. If so, it doesn't update anythign
    #      (i.e. doesn't make the guess)
    #
    #     """
    #
    #     # If the game is over, we can't make any guesses
    #     if self.done:
    #         return board, 0, True, board_guess_count
    #
    #     # If the board is complete, we can't make any guesses
    #     elif board_win:
    #
    #         return board, 0, True, board_guess_count
    #
    #     else:
    #
    #         # Grab action
    #         decoded_action = self.valid_words[action]
    #         encoded_action = self._encode(decoded_action)
    #
    #         # Update letters grid
    #         board['letters'][board_guess_count] = encoded_action
    #
    #         # Update colors
    #         new_colors = []
    #         for answer_letters, guess_letter in zip(encoded_answer, encoded_action):
    #
    #             # Green letter (i.e. correct letter in correct spot)
    #             if guess_letter == answer_letters:
    #                 new_colors.append(2)
    #                 if guess_letter not in self.green_letters:
    #                     self.green_letters.append(guess_letter)
    #
    #             # Yellow letter (i.e. correct letter in incorrect spot)
    #             elif guess_letter in encoded_answer:
    #                 new_colors.append(1)
    #                 if guess_letter not in self.green_letters:
    #                     self.yellow_letters.append(guess_letter)
    #
    #             # Gray letter (i.e. incorrect letter)
    #             else:
    #                 new_colors.append(0)
    #
    #         # Insert new color records
    #         board['colors'][board_guess_count] = new_colors
    #
    #
    #         # Compute board reward
    #         board_reward, board_win = self.compute_single_board_reward(board, board_guess_count)
    #
    #
    #         # Increment guess count on that board
    #         board_guess_count += 1
    #
    #         return board, board_reward, board_win, board_guess_count
    #
    # def step(self, action: int):
    #
    #     """
    #     Updates each board using the same action and then checks rewards/increments state/etc
    #
    #     action: integer for the index of self.valid_words selected as the guess
    #     """
    #
    #     # Convert the 1d state into grids for easier use
    #     state_list = self._convert_state_to_grids(self.state)
    #
    #     # Initialize stats for updating all boards to track everything
    #     # these will update the global attributes later
    #     step_boards = []
    #     step_rewards = []
    #     step_wins = []
    #     step_board_guess_counts = []
    #     for idx, board in enumerate(state_list):
    #
    #         # Get board answer and whether we've finished this board
    #         encoded_answer = self.encoded_answers[idx]
    #         board_win = self.wins[idx]
    #         board_guess_count = self.board_guess_counts[idx]
    #
    #
    #         # Update each board
    #         new_board, board_reward, win, board_guess_count = self.update_single_board(board,
    #                                                                                    action,
    #                                                                                    encoded_answer,
    #                                                                                    board_guess_count,
    #                                                                                    board_win)
    #
    #
    #         # Append step information
    #         step_boards.append(new_board)
    #         step_rewards.append(board_reward)
    #         step_wins.append(win)
    #         step_board_guess_counts.append(board_guess_count)
    #
    #
    #     # Compute total reward as mean of reward across n_boards. Picked mean so it's same reward across envs
    #     reward = np.mean(step_rewards)
    #
    #     # Increment guess count
    #     self.guess_count += 1
    #
    #
    #     # Check stopping conditions
    #     win = np.all(step_wins)
    #     reached_max_guesses = self.guess_count == self.n_guesses
    #     self.done = bool((reached_max_guesses) or (win))
    #
    #     # Update info
    #     self.wins = step_wins
    #     self.board_guess_counts = step_board_guess_counts
    #
    #     self.info = {'guess_count': self.guess_count,
    #                  'boards_win': self.wins,
    #                  'board_guess_counts': self.board_guess_counts,
    #                  'win': win}
    #
    #     # Convert grids back to 1d state
    #     self.state = self._convert_grids_to_state(step_boards)
    #
    #     return self.state, reward, self.done, self.info
    #
    # def reset(self, seed = None, return_info = False):
    #
    #     """
    #     Resets the environment
    #
    #     seed: random seed for selecting random answers. If none, the random answers won't be reproducible
    #     return_info: boolean for whether we should return the info from the last game
    #     """
    #
    #     # Create answers. We can keep answers as well.
    #     if self.keep_answers_on_reset:
    #         self.answers = self.answers
    #         self.encoded_answers = [self._encode(answer) for answer in self.answers]
    #     else:
    #         self.answers = np.random.choice(self.valid_answers, self.n_boards).tolist()
    #         self.encoded_answers = [self._encode(answer) for answer in self.answers]
    #
    #     # Initialize State
    #     self.state = -1 * np.ones(self.obs_dims, dtype = int)
    #
    #     # Initialize tracking variables
    #     self.guess_count = 0
    #     self.wins = [False] * self.n_boards # tracks which boards have been won already
    #     self.done = False # overall is the game done
    #     self.board_guess_counts = [self.guess_count] * self.n_boards # guess count for each board
    #
    #     # reset green and yellow trackers
    #     self.green_letters = []
    #     self.yellow_letters = []
    #
    #
    #     if return_info:
    #         return self.state, self.info
    #
    #     return self.state