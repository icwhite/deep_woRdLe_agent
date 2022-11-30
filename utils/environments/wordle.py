import numpy as np
from english_words import english_words_set
import gym
from gym import spaces
from gym.utils.env_checker import check_env

class Wordle(gym.Env): 
    
    def __init__(self, 
                 n_boards: int = 1, 
                 n_letters: int = 5,
                 n_guesses: int = 6, 
                 answers: list = None, 
                 seed: int = None, 
                 keep_answers_on_reset: bool = True): 
        
        """
        n_boards: number of boards that are played at once
        n_letters: number of letters per word
        n_guesses: number of guesses per board
        answers: list of answers. If None then answers are selected at random
        seed: seed for selecting random answers. If None then different answers will be selected each time
        keep_answers_on_reset: whether we should select new answers on each reset.
        """
        
        
        # Store attributes 
        self.n_boards = n_boards
        self.n_letters = n_letters
        self.n_guesses = n_guesses
        self.seed = seed 
        self.keep_answers_on_reset = keep_answers_on_reset
        
        # Create the list of valid words of length n_letters
        self.valid_words = [word.lower() for word in english_words_set if len(word) == self.n_letters]
        self.valid_words = [word for word in self.valid_words if "'" not in word and "." not in word and "&" not in word]
        self.valid_words = sorted(self.valid_words)

        
        # Create answers. If we pass a list it will set as answers. Otherwise it will generate a list 
        # of size n_boards of random answers. 
        # Alternatively, we can pass a seed for reproducibility
        if answers is not None: 
            assert(isinstance(answers, list))
            assert(len(answers) == self.n_boards)
            assert([len(answer) == self.n_letters for answer in answers])
            self.answers = answers
        else: 
            self.answers = np.random.choice(self.valid_words, self.n_boards).tolist()
            
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
        
    def _encode(self, word: str): 
        """
        Inputs a word of length n_letters as a string and maps to a list of ints corresponding to 
        each letter a = 0, b = 1, ..., z = 25
        """
        self.encoder = dict(zip(list("abcdefghijklmnopqrstuvwxyz"), np.arange(26)))
        return [self.encoder[letter] for letter in word]    
    
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
        
    def compute_single_board_reward(self, board: dict, board_guess_count: int): 
        
        """
        Because green is 2, the max score is 2 x number of letters
        We give a score of -1 for any bad guessses
        
        board: dict of form {'letters': [...], 'colors': [...]}
        """
        score = np.sum(board['colors'][board_guess_count])
        max_score = 2 * self.n_letters
        reward = 1 if score == max_score else -1
        
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

            # Update colors 
            new_colors = []
            for answer_letters, guess_letter in zip(encoded_answer, encoded_action): 

                # Green letter (i.e. correct letter in correct spot)
                if guess_letter == answer_letters: 
                    new_colors.append(2)

                # Yellow letter (i.e. correct letter in incorrect spot)
                elif guess_letter in encoded_answer: 
                    new_colors.append(1)

                # Gray letter (i.e. incorrect letter)
                else: 
                    new_colors.append(0)

            # Insert new color records
            board['colors'][board_guess_count] = new_colors


            # Compute board reward 
            board_reward = self.compute_single_board_reward(board, board_guess_count)


            # Check if we've won this board now
            board_win = board_reward > 0
            
            # Increment guess count on that board
            board_guess_count += 1
            
            return board, board_reward, board_win, board_guess_count
    
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

        return self.state, reward, self.done, self.info
    
    def reset(self, seed = None, return_info = False):
        
        """
        Resets the environment
        
        seed: random seed for selecting random answers. If none, the random answers won't be reproducible
        return_info: boolean for whether we should return the info from the last game
        """
    
        # Create answers. We can keep answers as well.
        if self.keep_answers_on_reset:
            self.answers = self.answers
            self.encoded_answers = [self._encode(answer) for answer in self.answers]
        else: 
            self.answers = np.random.choice(self.valid_words, self.n_boards).tolist()
            self.encoded_answers = [self._encode(answer) for answer in self.answers]
                        
        # Initialize State
        self.state = -1 * np.ones(self.obs_dims, dtype = int)
        
        # Initialize tracking variables
        self.guess_count = 0
        self.wins = [False] * self.n_boards # tracks which boards have been won already
        self.done = False # overall is the game done
        self.board_guess_counts = [self.guess_count] * self.n_boards # guess count for each board
        
    
        if return_info:
            return self.state, self.info
                        
        return self.state