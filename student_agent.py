# Remember to adjust your student ID in meta.xml
import gc
import pickle
import random
import gym
import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import joblib

from gym import spaces
from collections import defaultdict
from functools import lru_cache

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

@lru_cache(maxsize=4096)
def process_row_cache(row_tuple):
    row = list(row_tuple)
    non_zero = [x for x in row if x != 0]
    merged = []
    score_gained = 0
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
            merged_val = non_zero[i] * 2
            merged.append(merged_val)
            score_gained += merged_val
            skip = True
        else:
            merged.append(non_zero[i])
    new_row = merged + [0] * (len(row) - len(merged))
    return tuple(new_row), score_gained

def process_np_row(row):
    new_row_tuple, score_gained = process_row_cache(tuple(row))
    new_row = np.array(new_row_tuple)
    return new_row, score_gained

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # 行動空間: 0: 上, 1: 下, 2: 左, 3: 右
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row, row_score = process_np_row(original_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, new_row):
                moved = True
                self.score += row_score
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = original_row[::-1]
            new_row, row_score = process_np_row(reversed_row)
            new_row = new_row[::-1]
            self.board[i] = new_row
            if not np.array_equal(original_row, new_row):
                moved = True
                self.score += row_score
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            new_col, col_score = process_np_row(original_col)
            self.board[:, j] = new_col
            if not np.array_equal(original_col, new_col):
                moved = True
                self.score += col_score
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = original_col[::-1]
            new_col, col_score = process_np_row(reversed_col)
            new_col = new_col[::-1]
            self.board[:, j] = new_col
            if not np.array_equal(original_col, new_col):
                moved = True
                self.score += col_score
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True
    
    
    def move(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        done = self.is_game_over()
        return self.board, self.score, done, {}

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"
        original_score = self.score

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved
        after_move_board = self.board.copy()
        reward = self.score - original_score

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return after_move_board, self.board.copy(), reward, done, {}

    def simulate_step(self, action):
        """
        Simulates taking an action without modifying the actual game state.
        """
        assert self.action_space.contains(action), "Invalid action"

        # Save current state
        original_board = self.board.copy()
        original_score = self.score

        # Apply the action
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        after_move_board = self.board.copy()
        reward = self.score - original_score

        if moved:
            self.add_random_tile()

        done = self.is_game_over()
        after_add_tile_board = self.board.copy()

        # Restore original state
        self.board = original_board
        self.score = original_score

        return after_move_board, after_add_tile_board, reward, done, {}
    
    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"Score: {self.score}"
        if action is not None:
            title += f" | Action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # 上
            for j in range(self.size):
                col = temp_board[:, j]
                new_col, _ = process_np_row(col)
                temp_board[:, j] = new_col
        elif action == 1:  # 下
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col, _ = process_np_row(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # 左
            for i in range(self.size):
                row = temp_board[i]
                new_row, _ = process_np_row(row)
                temp_board[i] = new_row
        elif action == 3:  # 右
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row, _ = process_np_row(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))

class NTupleNetwork:
    def __init__(self, vinit=0):
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

        self.weights = [defaultdict(float) for _ in range(3)]
        self.max_tile_log = 15
        self.vinit = vinit
        if self.vinit > 0:
            self.apply_optimistic_initialization()
    
    def apply_optimistic_initialization(self):
        init_value = self.vinit / len(self.base_patterns)
        
        for stage in range(3):
            self.weights[stage] = defaultdict(lambda: init_value)
        
    def _generate_all_patterns(self):
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
        tile_values = [int(math.log2(board[pos[0], pos[1]]) if board[pos[0], pos[1]] > 0 else 0) for pos in pattern_pos]
        
        index = 0
        for value in tile_values:
            index = index * (self.max_tile_log + 1) + value
        
        return index
    
    def evaluate(self, board, stage=0):
        """Evaluate a board state using the N-tuple network for a specific stage"""
        values = []
        
        for i, pattern_pos in enumerate(self.all_patterns):
            index = self.get_tuple_index(board, pattern_pos)
            values.append(self.weights[stage][(i//self.symmetric_pattern_cnt, index)])
        
        return np.sum(values)
    
    def update(self, board, target, stage=0, alpha=0.01):
        current_value = self.evaluate(board, stage)
        td_error = target - current_value

        for i, pattern_pos in enumerate(self.all_patterns):
            index = self.get_tuple_index(board, pattern_pos)
            self.weights[stage][i//self.symmetric_pattern_cnt, index] += alpha * td_error / len(self.base_patterns)
        
        return td_error
    
    def save_weights(self, filename):
        serializable_weights = []
        for stage_weights in self.weights:
            serializable_weights.append(dict(stage_weights))

        with open(f"models/{filename}", "wb") as f:
            pickle.dump(serializable_weights, f)

    def load_weights(self, filename):
        print(f"Loading weights from {filename}...")
        serializable_weights = joblib.load(filename)
        # with open(f"{filename}", "rb") as f:
        #     serializable_weights = pickle.load(f)
        for i, stage_weights in enumerate(serializable_weights):
            self.weights[i] = defaultdict(float, stage_weights)
        
        print(f"Loaded weights from {filename} successfully.")

class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        self.default_value = 0.0

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0      


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99, debug=False):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.debug = debug

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        max_UCB_value = -float('inf')
        best_child = None

        if self.debug:
            print("--- Selecting Node ---")
        for child in node.children.values():
            if child.visits == 0:
                return child

            exploitation = child.total_reward / child.visits
            exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
            UCB_value = exploitation + exploration
            if self.debug:
                print(f"Child total_reward: {child.total_reward}, visits: {child.visits}, node.visits: {node.visits}")
                print(f"Exploitation: {exploitation}, Exploration: {exploration}, Score: {UCB_value}")

            if UCB_value > max_UCB_value:
                max_UCB_value = UCB_value
                best_child = child

        return best_child
    
    def rollout(self, sim_env, depth):
        # Perform a random rollout until reaching the maximum depth or a terminal state
        # Use the approximator to evaluate the final state
        prev_score = sim_env.score        
        for d in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                return 0
            else:
                best_action = random.choice(legal_moves)
                # Simulate the action                        
                _, next_state, reward, done, _ = sim_env.step(best_action)
            
        value = self.approximator.value(next_state)
        score = sim_env.score - prev_score
        if self.debug:
            print(f"Score: {score}, Value: {value}")
        return score + value

    def backpropagate(self, node, reward):
        # Propagate the obtained reward back up the tree
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            action = node.untried_actions.pop()
            _, next_state, reward, done, _ = sim_env.step(action)
            child_node = TD_MCTS_Node(sim_env.board, sim_env.score, parent=node, action=action)
            node.children[action] = child_node
            node = child_node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)
        end(root)

    def best_action(self, root):
        best_visits = -1
        best_action = None
        
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
                
        return best_action

gc.collect()
env = Game2048Env()
network = NTupleNetwork()
network.load_weights("converted_stage1_weights_ep55000_new.pkl")
class ValueApproximator:
    def __init__(self, network):
        self.network = network
        
    def value(self, board):
        return self.network.evaluate(board, 0)

approximator = ValueApproximator(network)
mcts = TD_MCTS(env, approximator, iterations=200, exploration_constant=1.0, rollout_depth=5, gamma=0.99, debug=False)

def init_model():
    global env, network, approximator, mcts
    if mcts is None:
        gc.collect()
        env = Game2048Env()
        network = NTupleNetwork()
        network.load_weights("converted_stage1_weights_ep55000_new.pkl")
        class ValueApproximator:
            def __init__(self, network):
                self.network = network
                
            def value(self, board):
                return self.network.evaluate(board, 0)

        approximator = ValueApproximator(network)
        mcts = TD_MCTS(env, approximator, iterations=1000, exploration_constant=1.0, rollout_depth=10, gamma=0.99, debug=False)

def get_action(state, score):
    init_model()
    global env, mcts
    env.board = state.copy()
    env.score = score
    
    root = TD_MCTS_Node(state.copy(), score)
    
    for _ in range(mcts.iterations):
        mcts.run_simulation(root)
    
    best_action = mcts.best_action(root)
    
    return best_action

def has_tile(board, tile_log):
    target_value = 2 ** tile_log
    return np.any(board == target_value)


def has_tiles(board, tile_logs):
    return all(has_tile(board, tile_log) for tile_log in tile_logs)


def val(state, score, action, depth=2):
    global env
    env.board = state.copy()
    env.score = score
    after_move_board, next_state, reward, done, _ = env.simulate_step(action)
    
    if done:
        return reward
    empty_cells = list(zip(*np.where(after_move_board == 0)))
    if not empty_cells:
        return reward + approximator.value(after_move_board)
    
    expected_value = 0
    total_probability = 0
    
    for cell in empty_cells:
        board_with_2 = after_move_board.copy()
        board_with_2[cell] = 2
        prob_2 = 0.9 / len(empty_cells)
        
        if depth > 1:
            best_value_2 = next_val(board_with_2, score + reward)
        else:
            best_value_2 = next_val(board_with_2, score + reward)
        
        expected_value += prob_2 * best_value_2
        total_probability += prob_2
        board_with_4 = after_move_board.copy()
        board_with_4[cell] = 4
        prob_4 = 0.1 / len(empty_cells)        
        
        if depth > 1:
            best_value_4 = next_val(board_with_4, score + reward)
        else:
            best_value_4 = next_val(board_with_4, score + reward)
            
        expected_value += prob_4 * best_value_4
        total_probability += prob_4
    
    if total_probability > 0:
        expected_value /= total_probability
    
    return expected_value
mcts.iterations = 1
mcts.rollout_depth = 1

def end(node):
    env = Game2048Env()
    env.board = node.state.copy()
    legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    node.children.clear()
    for action in legal_actions:
        _, next_state, reward, done, _ = env.simulate_step(action)
        child_node = TD_MCTS_Node(next_state, env.score+reward, parent=node, action=action)
        child_node.visits = val(node.state, node.score, action)
        node.children[action] = child_node

def next_val(board, score):
    global env
    env.board = board.copy()
    env.score = score
    
    legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_actions:
        return 0
    
    best_value = float('-inf')
    
    for action in legal_actions:
        after_move_board, _, action_reward, done, _ = env.simulate_step(action)
        value = approximator.value(after_move_board)
        if value > best_value:
            best_value = value
    
    return best_value

def main():
    """
    Main function to run the 2048 game using the MCTS agent.
    """
    main_env = Game2048Env()
    final_scores = []

    for i in range(10):
        state = main_env.reset()
        done = False
        total_reward = 0
        turn = 0
        
        while not done:
            turn += 1
            # Get action from the agent
            action = get_action(state, main_env.score)
            
            # Take the action
            _, next_state, reward, done, _ = main_env.step(action)
            
            total_reward += reward
            state = next_state
            
            if turn % 100 == 0:
                print(f"Game {i+1}, Turn: {turn}, Score: {main_env.score}")
            
            if done:
                print(f"Game {i+1} Over! Final Score: {main_env.score}")
                # Find the highest tile
                highest_tile = np.max(main_env.board)
                print(f"Highest tile achieved: {highest_tile}")
                final_scores.append(main_env.score)

    print(f"Average Score over {len(final_scores)} games: {np.mean(final_scores)}")
    print(f"Max Score over {len(final_scores)} games: {np.max(final_scores)}")
    print(f"Min Score over {len(final_scores)} games: {np.min(final_scores)}")
    print(f"Final Scores: {final_scores}")
    plt.plot(final_scores)
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.title("Scores over 10 Games")
    plt.show()

if __name__ == "__main__":
    main()