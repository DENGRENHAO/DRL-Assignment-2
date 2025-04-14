import os
import gc
import copy
import math
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

from student_agent import Game2048Env

class NTupleNetwork:
    def __init__(self, vinit=0):
        # Define the N-tuples (patterns) as described in the paper Fig. 4(b)
        # These are the base patterns. We'll generate rotations and mirrors
        self.base_patterns = [
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2)],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
            [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (2, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)],
        ]
        
        self._generate_all_patterns()
        self.symmetric_pattern_cnt = len(self.all_patterns) // len(self.base_patterns)

        # self._print_all_patterns()
        
        # Initialize feature weights for each stage
        # self.weights[stage][(f"pattern{n}", index)] = weight
        self.weights = [defaultdict(float) for _ in range(3)]
        
        # Maximum tile value (log base 2) for indexing
        # For 2048 game, tiles can be 2, 4, 8, ..., 32768 = 2^15
        self.max_tile_log = 15
        
        # For large-tile features
        self.large_tile_thresholds = [11, 12, 13, 14, 15]  # Log2 values for 2048, 4096, 8192, 16384, 32768

        self.vinit = vinit
        if self.vinit > 0:
            self.apply_optimistic_initialization()
    
    def apply_optimistic_initialization(self):
        """Apply optimistic initialization to all weights"""
        # Set all weights to vinit/m where m is the number of features
        init_value = self.vinit / len(self.base_patterns)
        
        # Initialize all weights optimistically for all stages
        for stage in range(3):
            self.weights[stage] = defaultdict(lambda: init_value)
    
    def _print_all_patterns(self):
        """Print all generated patterns with 4x4 grid representation"""
        for i, pattern in enumerate(self.all_patterns):
            if i % self.symmetric_pattern_cnt == 0:
                print(f"Pattern {i // self.symmetric_pattern_cnt + 1}")
            grid = np.zeros((4, 4), dtype=int)
            for x, y in pattern:
                grid[x, y] = 1
            print(grid)
        
    def _generate_all_patterns(self):
        """Generate all rotations and mirrors of the base tuples"""
        def rot90(pattern):
            return [(y, 3 - x) for x, y in pattern]

        def rot180(pattern):
            return [(3 - x, 3 - y) for x, y in pattern]

        def rot270(pattern):
            return [(3 - y, x) for x, y in pattern]

        def flip_horizontal(pattern):
            return [(x, 3 - y) for x, y in pattern]
        
        self.all_patterns = []
        for base_pattern in self.base_patterns:
            self.all_patterns.append(base_pattern)
            self.all_patterns.append(rot90(base_pattern))
            self.all_patterns.append(rot180(base_pattern))
            self.all_patterns.append(rot270(base_pattern))
            self.all_patterns.append(flip_horizontal(base_pattern))
            self.all_patterns.append(flip_horizontal(rot90(base_pattern)))
            self.all_patterns.append(flip_horizontal(rot180(base_pattern)))
            self.all_patterns.append(flip_horizontal(rot270(base_pattern)))
    
    def get_tuple_index(self, board, pattern_pos):
        """Convert a tuple pattern on a board to an index"""
        # Extract tile values for the given positions and convert to log2
        tile_values = [int(math.log2(board[pos[0], pos[1]]) if board[pos[0], pos[1]] > 0 else 0) for pos in pattern_pos]
        
        # Compute index (base-conversion: each position can have 0-15 value)
        index = 0
        for value in tile_values:
            index = index * (self.max_tile_log + 1) + value
        
        return index
    
    def get_large_tile_index(self, board):
        """Get the index for large-tile features (combinations of tiles >= 2048)"""
        # Count occurrences of each large tile
        counts = [0] * len(self.large_tile_thresholds)
        
        for i in range(4):
            for j in range(4):
                if board[i, j] > 0:
                    log_val = int(math.log2(board[i, j]))
                    for k, threshold in enumerate(self.large_tile_thresholds):
                        if log_val == threshold:
                            counts[k] += 1
        
        # Compute a unique index for this combination
        # Since we don't expect many large tiles, we can use a simple encoding
        # Maximum expected count per large tile is 1 (for 32768) or 2 (for others)
        # So we use 2 bits per count (can represent 0, 1, 2, 3)
        index = 0
        for count in counts:
            index = (index * 4) + min(count, 3)  # Limit to max 3 for safety
        
        return index
    
    def evaluate(self, board, stage=0):
        """Evaluate a board state using the N-tuple network for a specific stage"""
        values = []
        
        for i, pattern_pos in enumerate(self.all_patterns):
            index = self.get_tuple_index(board, pattern_pos)
            values.append(self.weights[stage][(i//self.symmetric_pattern_cnt, index)])
        
        # Add large-tile feature weight
        # large_index = self.get_large_tile_index(board)
        # if large_index > 0:
        #     values.append(self.weights[stage][('large', large_index)])
        
        # return np.mean(values)
        return np.sum(values)
    
    def update(self, board, target, stage=0, alpha=0.01):
        """Update weights based on the current board and target value"""
        current_value = self.evaluate(board, stage)
        td_error = target - current_value

        # print(f"Current Value: {current_value}, Target: {target}, alpha * TD Error: {alpha * td_error}")
        
        # Update weights for all pattern tuples
        for i, pattern_pos in enumerate(self.all_patterns):
            index = self.get_tuple_index(board, pattern_pos)
            # print(f"Before Update: {self.weights[stage][(f'pattern{i//self.symmetric_pattern_cnt}', index)]}")
            self.weights[stage][(i//self.symmetric_pattern_cnt, index)] += alpha * td_error / len(self.all_patterns)
            # print(f"After Update: {self.weights[stage][(f'pattern{i//self.symmetric_pattern_cnt}', index)]}")
        
        # Update large-tile feature weight
        # large_index = self.get_large_tile_index(board)
        # if large_index > 0:
        #     self.weights[stage][('large', large_index)] += alpha * td_error
        
        return td_error
    
    # self.weights[stage][({n}", idex)] = weit
    def save_weights(self, filename):
        """Save the network weights to a file"""
        serializable_weights = []
        for stage_weights in self.weights:
            serializable_weights.append(dict(stage_weights))

        with open(f"models/{filename}", "wb") as f:
            pickle.dump(serializable_weights, f)

    def load_weights(self, filename):
        """Load the network weights from a file"""
        with open(f"models/{filename}", "rb") as f:
            serializable_weights = pickle.load(f)

        for i, stage_weights in enumerate(serializable_weights):
            self.weights[i] = defaultdict(float, stage_weights)
                

class ThreeStageTDLearning:
    def __init__(self, alpha=0.1, gamma=1.0, exploration=0.1, window_size=1000, 
                 weight_saving_interval=1000, search_depth=2, vinit=0):
        self.env = Game2048Env()
        self.network = NTupleNetwork(vinit=vinit)
        self.orig_alpha = alpha
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor (typically 1.0 for 2048)
        self.exploration = exploration  # Exploration rate
        self.window_size = window_size
        self.weight_saving_interval = weight_saving_interval
        self.search_depth = search_depth
        
        # Thresholds for stage transitions (log2 values)
        # self.t_16k = 14  # First 16384-tile (2^14)
        # self.t_16k_8k = (14, 13)  # Both 16384 (2^14) and 8192 (2^13) tiles
        
        self.t_16k = 12  # First 16384-tile (2^14)
        self.t_16k_8k = (12, 11)  # Both 16384 (2^14) and 8192 (2^13) tiles

        # Collected boards for stage transitions
        self.stage2_boards = []
        self.stage3_boards = []
        self.stage_board_limit = 100000  # Limit on collected boards for stage transitions
        
        # Statistics
        self.scores_history = []
        self.max_tiles_history = []
    
    def has_tile(self, board, tile_log):
        """Check if the board has a tile of value 2^tile_log"""
        target_value = 2 ** tile_log
        return np.any(board == target_value)
    
    def has_tiles(self, board, tile_logs):
        """Check if the board has all the specified tiles"""
        return all(self.has_tile(board, tile_log) for tile_log in tile_logs)
    
    def select_action(self, stage=0):
        """Select action using epsilon-greedy policy"""
        legal_actions = [a for a in range(4) if self.env.is_move_legal(a)]
        if not legal_actions:
            return 0
        
        # if random.random() < self.exploration:
        #     # Random exploration
        #     return random.choice(legal_actions)
        
        # Greedy action selection
        best_value = float('-inf')
        best_action = 0
        
        for action in legal_actions:
            # Create a copy of the env to simulate the action
            after_state, next_state, reward, done, _ = self.env.simulate_step(action)
            
            if done:
                # Immediate reward for game-over states
                value = reward
            else:
                # Expectimax: Consider all possible random tile additions
                value = reward + self.expectimax(after_state.copy(), 0, stage, is_max_node=False)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

    def expectimax(self, board, depth, stage, is_max_node=True):
        """
        Perform expectimax search to evaluate board states
        
        Parameters:
        - board: The current board state
        - depth: Current depth in the search tree
        - stage: Which stage weights to use
        - is_max_node: True if it's the player's turn, False if it's the random tile placement
        
        Returns:
        - Expected value of the board state
        """
        # Terminal condition: max depth reached or game over
        if depth >= self.search_depth:
            return self.network.evaluate(board, stage)
        
        if is_max_node:
            # Max node (player's turn)
            best_value = float('-inf')
            
            # Create temporary environment to check legal moves
            temp_env = copy.deepcopy(self.env)
            temp_env.board = board.copy()
            legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
            
            if not legal_actions:
                # No legal moves, return current evaluation
                return self.network.evaluate(board, stage)
            
            for action in legal_actions:
                # Simulate the action
                action_env = copy.deepcopy(temp_env)
                after_state, next_state, reward, done, _ = action_env.step(action)
                
                if done:
                    value = reward
                else:
                    # Next level is a chance node
                    value = reward + self.expectimax(after_state.copy(), depth + 1, stage, False)
                
                best_value = max(best_value, value)
            
            return best_value
        
        else:
            # Chance node (random tile placement)
            # Get all empty cells
            empty_cells = list(zip(*np.where(board == 0)))
            if not empty_cells:
                # No empty cells, game is over
                return self.network.evaluate(board, stage)
            
            # Consider all possible tile placements
            expected_value = 0
            
            # For efficiency, we can sample a subset of empty cells if there are too many
            # This is especially helpful in the early game where many cells are empty
            sample_size = min(len(empty_cells), 4)  # Adjustable parameter
            if len(empty_cells) > sample_size:
                empty_cells = random.sample(empty_cells, sample_size)
            
            # Probability of a 2 (90%) and a 4 (10%)
            prob_2 = 0.9 / len(empty_cells)
            prob_4 = 0.1 / len(empty_cells)
            
            for x, y in empty_cells:
                # Try placing a 2
                board_with_2 = board.copy()
                board_with_2[x, y] = 2
                
                # Next level is a max node
                value_2 = self.expectimax(board_with_2, depth + 1, stage, True)
                expected_value += prob_2 * value_2
                
                # Try placing a 4
                board_with_4 = board.copy()
                board_with_4[x, y] = 4
                
                # Next level is a max node
                value_4 = self.expectimax(board_with_4, depth + 1, stage, True)
                expected_value += prob_4 * value_4
            
            return expected_value
    
    def determine_stage(self, board):
        """Determine which stage the game is in based on the board"""
        if self.has_tiles(board, self.t_16k_8k):
            return 2  # Stage 3
        elif self.has_tile(board, self.t_16k):
            return 1  # Stage 2
        else:
            return 0  # Stage 1

    def learn_from_trajectory(self, trajectory, stage=0):
        """Learn from trajectory using LEARN EVALUATION method shown in Figure 6"""
        for i in range(len(trajectory) - 1, -1, -1):
            state, after_state, next_state, reward, done = trajectory[i]
            
            if done:
                # Terminal state - use just the reward
                target = reward
                self.network.update(after_state.copy(), target, stage=stage, alpha=self.alpha)
            else:
                # For non-terminal states, get the next best action
                self.env.board = next_state.copy()
                next_action = self.select_action(stage=stage)
                next_after_state, _, next_reward, next_done, _ = self.env.simulate_step(next_action)
                
                # Calculate target using afterstate evaluation
                target = next_reward + self.gamma * self.network.evaluate(next_after_state.copy(), stage=stage)
                # print(f"next_reward: {next_reward}, next_after_state value: {self.network.evaluate(next_after_state, stage=stage)}")
                
                # Update the value function for the after_state
                self.network.update(after_state.copy(), target, stage=stage, alpha=self.alpha)
    
    def train_stage1(self, num_episodes=5000000, start_episode=0):
        if start_episode > 0:
            print(f"Resuming training from episode {start_episode}...")
            # Load weights from the specified episode
            self.network.load_weights(f"stage1_weights_ep{start_episode}_new.pkl")
        """Train the first stage until saturation"""
        print("Training Stage 1...")
        
        # Monitoring variables for saturation detection
        score_windows = []
        progress_bar = tqdm(range(start_episode, num_episodes))
        
        # Calculate episode numbers for learning rate changes (50% and 75% of training)
        half_point = num_episodes // 2
        three_quarters_point = num_episodes * 3 // 4
        
        for episode in progress_bar:
            # Update learning rate according to schedule
            if episode >= three_quarters_point:
                self.alpha = 0.001  # 75% and beyond: alpha = 0.001
            elif episode >= half_point:
                self.alpha = 0.01   # 50-75%: alpha = 0.01
            else:
                self.alpha = 0.1    # 0-50%: alpha = 0.1
            
            # Initialize game
            state = self.env.reset()
            done = False
            added = False
            
            # Initialize trajectory for this episode
            trajectory = []
            
            while not done:
                # Select and perform action
                action = self.select_action(stage=0)  # Always use stage 0 weights
                after_state, next_state, reward, done, _ = self.env.step(action)
                
                # Store experience in trajectory
                trajectory.append((state.copy(), after_state.copy(), next_state.copy(), reward, done))
                
                if not added and self.determine_stage(next_state.copy()) == 1:
                    # First time a 16384-tile appears - collect for stage 2
                    self.stage2_boards.append((next_state.copy(), self.env.score))
                    if len(self.stage2_boards) >= self.stage_board_limit:
                        self.stage2_boards = self.stage2_boards[-self.stage_board_limit:]
                    added = True
                
                state = next_state.copy()
            
            # Learn from trajectory using TD(0) with afterstate evaluation
            self.learn_from_trajectory(trajectory, stage=0)
            
            # Record statistics
            self.scores_history.append(self.env.score)
            max_tile = np.max(state)
            self.max_tiles_history.append(max_tile)
            
            # Check for saturation (every window_size episodes)
            if (episode + 1) % self.window_size == 0:
                avg_score = np.mean(self.scores_history[-self.window_size:])
                score_windows.append(avg_score)
                progress_bar.set_description(f"Episode {episode+1}, Avg Score: {avg_score:.3f}, Max Tile: {max_tile}, Alpha: {self.alpha}, Len Stage 2 Boards: {len(self.stage2_boards)}")
                # Plot learning curve
                self.plot_learning_curve(stage=1, episode=episode+1)

                # Save checkpoints
                if (episode + 1) % self.weight_saving_interval == 0:
                    self.network.save_weights(f"stage1_weights_ep{episode+1}_new.pkl")
                
        
        # Save final weights
        self.network.save_weights("stage1_weights_final.pkl")
        
        print(f"Stage 1 training complete. Collected {len(self.stage2_boards)} boards for Stage 2.")
    
    def train_stage2(self, num_episodes=5000000):
        """Train the second stage weights using collected boards"""
        print("Training Stage 2...")
        
        if not self.stage2_boards:
            print("No boards collected for Stage 2. Run Stage 1 training first.")
            return
        
        half_point = num_episodes // 2
        three_quarters_point = num_episodes * 3 // 4
        
        score_windows = []
        progress_bar = tqdm(range(num_episodes))
    
        for episode in progress_bar:
            # Update learning rate according to schedule
            if episode >= three_quarters_point:
                self.alpha = 0.001  # 75% and beyond: alpha = 0.001
            elif episode >= half_point:
                self.alpha = 0.01   # 50-75%: alpha = 0.01
            else:
                self.alpha = 0.1    # 0-50%: alpha = 0.1

            # Start from a collected board
            state, score = random.choice(self.stage2_boards)
            self.env.board = state.copy()
            self.env.score = score
            done = False
            added = False

            # Initialize trajectory for this episode
            trajectory = []
            
            while not done:
                # Select and perform action
                action = self.select_action(stage=1)  # Always use stage10 weights
                after_state, next_state, reward, done, _ = self.env.step(action)
                
                # Store experience in trajectory
                trajectory.append((state.copy(), after_state.copy(), next_state.copy(), reward, done))
                
                if not added and self.determine_stage(next_state.copy()) == 2:
                    # collect for stage 3
                    self.stage3_boards.append((next_state.copy(), self.env.score))
                    if len(self.stage3_boards) >= self.stage_board_limit:
                        self.stage3_boards = self.stage3_boards[-self.stage_board_limit:]
                    
                    added = True
                
                state = next_state.copy()
            
            # Learn from trajectory using TD(0) with afterstate evaluation
            self.learn_from_trajectory(trajectory, stage=1)
            
            # while not done:
            #     # Select and perform action
            #     action = self.select_action(stage=1)  # Use stage 1 weights
            #     after_state, next_state, reward, done, _ = self.env.step(action)
                
            #     # TD update
            #     if not done:
            #         target = reward + self.gamma * self.network.evaluate(after_state, stage=1)
            #     else:
            #         target = reward
                
            #     self.network.update(state, target, stage=1, alpha=self.alpha)
                
            #     # Collect boards for stage transitions
            #     if self.determine_stage(next_state) == 2:
            #         # Both 16384 and 8192 tiles appear - collect for stage 3
            #         self.stage3_boards.append((next_state.copy(), self.env.score))
            #         if len(self.stage3_boards) >= self.stage_board_limit:
            #             self.stage3_boards = self.stage3_boards[-self.stage_board_limit:]
                
            #     state = next_state
            
            # Record statistics
            self.scores_history.append(self.env.score)
            max_tile = np.max(state)
            self.max_tiles_history.append(max_tile)

            # Check for saturation (every window_size episodes)
            if (episode + 1) % self.window_size == 0:
                avg_score = np.mean(self.scores_history[-self.window_size:])
                score_windows.append(avg_score)
                progress_bar.set_description(f"Episode {episode+1}, Avg Score: {avg_score:.3f}, Max Tile: {max_tile}, Alpha: {self.alpha}, Len Stage 3 Boards: {len(self.stage3_boards)}")
                
                # Plot learning curve
                self.plot_learning_curve(2)

                # Save checkpoints
                if (episode + 1) % self.weight_saving_interval == 0:
                    self.network.save_weights(f"stage2_weights_ep{episode+1}.pkl")
        
        # Save final weights
        self.network.save_weights("stage2_weights_final.pkl")
        
        print(f"Stage 2 training complete. Collected {len(self.stage3_boards)} boards for Stage 3.")
    
    def train_stage3(self, num_episodes=5000000):
        """Train the third stage weights using collected boards"""
        print("Training Stage 3...")
        
        if not self.stage3_boards:
            print("No boards collected for Stage 3. Run Stage 2 training first.")
            return
        
        half_point = num_episodes // 2
        three_quarters_point = num_episodes * 3 // 4
        
        score_windows = []
        progress_bar = tqdm(range(num_episodes))
    
        for episode in progress_bar:
            # Update learning rate according to schedule
            if episode >= three_quarters_point:
                self.alpha = 0.001  # 75% and beyond: alpha = 0.001
            elif episode >= half_point:
                self.alpha = 0.01   # 50-75%: alpha = 0.01
            else:
                self.alpha = 0.1    # 0-50%: alpha = 0.1
                
            # Start from a collected board
            state, score = random.choice(self.stage3_boards)
            self.env.board = state.copy()
            self.env.score = score
            done = False

            # Initialize trajectory for this episode
            trajectory = []
            
            while not done:
                # Select and perform action
                action = self.select_action(stage=2)  # Always use stage 3 weights
                after_state, next_state, reward, done, _ = self.env.step(action)
                
                # Store experience in trajectory
                trajectory.append((state.copy(), after_state.copy(), next_state.copy(), reward, done))
                
                state = next_state.copy()
            
            # Learn from trajectory using TD(0) with afterstate evaluation
            self.learn_from_trajectory(trajectory, stage=2)
            
            # while not done:
            #     # Select and perform action
            #     action = self.select_action(stage=2)  # Use stage 2 weights
            #     after_state, next_state, reward, done, _ = self.env.step(action)
                
            #     # TD update
            #     if not done:
            #         target = reward + self.gamma * self.network.evaluate(after_state, stage=2)
            #     else:
            #         target = reward
                
            #     self.network.update(state, target, stage=2, alpha=self.alpha)
                
            #     state = next_state
            
            # Record statistics
            self.scores_history.append(self.env.score)
            max_tile = np.max(state)
            self.max_tiles_history.append(max_tile)

            # Check for saturation (every window_size episodes)
            if (episode + 1) % self.window_size == 0:
                avg_score = np.mean(self.scores_history[-self.window_size:])
                score_windows.append(avg_score)
                progress_bar.set_description(f"Episode {episode+1}, Avg Score: {avg_score:.3f}, Max Tile: {max_tile}")
                
                # Plot learning curve
                self.plot_learning_curve(3)

                # Save checkpoints
                if (episode + 1) % self.weight_saving_interval == 0:
                    self.network.save_weights(f"stage3_weights_ep{episode+1}.pkl")
        
        # Save final weights
        self.network.save_weights("stage3_weights_final.pkl")
        
        print("Stage 3 training complete.")
    
    def play_game_with_multistage(self, render=False):
        """Play a game using the multi-stage strategy with expectimax search"""
        state = self.env.reset()
        done = False
        current_stage = -1
        
        while not done:
            # Determine the current stage
            new_stage = self.determine_stage(state)
            if new_stage != current_stage:
                current_stage = new_stage
                print(f"Switching to Stage {current_stage + 1}")
            
            # Select best action using appropriate stage weights
            action = self.select_action(stage=current_stage)
            
            if render:
                self.env.render(action=action)
                time.sleep(0.5)
            
            # Execute action
            _, state, _, done, _ = self.env.step(action)
        
        print(f"Game Over! Score: {self.env.score}, Max Tile: {np.max(state)}")
        return self.env.score, np.max(state)

    def evaluate_agent(self, num_games=100):
        """Evaluate the agent by playing multiple games with specified search depth"""
        scores = []
        max_tiles = []
        
        print(f"Evaluating agent with {self.search_depth}-ply expectimax search...")
        
        for _ in tqdm(range(num_games)):
            score, max_tile = self.play_game_with_multistage(render=False)
            scores.append(score)
            max_tiles.append(max_tile)
        
        # Print statistics
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores)}")
        print(f"Tile Statistics:")
        
        # Count occurrences of each max tile
        max_tile_counts = {}
        for tile in max_tiles:
            max_tile_counts[tile] = max_tile_counts.get(tile, 0) + 1
        
        for tile in sorted(max_tile_counts.keys()):
            percentage = (max_tile_counts[tile] / num_games) * 100
            print(f"  {tile}: {max_tile_counts[tile]} games ({percentage:.3f}%)")
            
        return scores, max_tiles
    
    def plot_learning_curve(self, stage, episode=None):
        """Plot learning curve for a specific stage"""
        # Calculate moving averages
        avg_scores = []
        for i in range(0, len(self.scores_history), self.window_size):
            window = self.scores_history[i:i+self.window_size]
            if window:
                avg_scores.append(np.mean(window))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(avg_scores)), avg_scores, label='Average Score', color='blue')
        plt.title(f'Stage {stage} Learning Curve')
        plt.xlabel('Episodes (x{})'.format(self.window_size))
        plt.ylabel('Average Score')
        plt.savefig(f'models/stage{stage}_learning_curve_new.png')
        plt.close()

# Create the trainer
trainer = ThreeStageTDLearning(alpha=0.1, gamma=1.0, exploration=0.1, window_size=10, weight_saving_interval=2500, search_depth=0, vinit=0)

# Train each stage
trainer.train_stage1(num_episodes=200000, start_episode=45000)
# trainer.train_stage2(num_episodes=20000)
# trainer.train_stage3(num_episodes=20000)

# Save all weights
trainer.network.save_weights("final_weights_all_stages.pkl")