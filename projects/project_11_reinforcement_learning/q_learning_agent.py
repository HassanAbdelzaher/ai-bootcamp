"""
Project 11: Reinforcement Learning
Q-learning agent for Grid World navigation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

print("=" * 70)
print("Project 11: Reinforcement Learning")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Grid World Environment
# ============================================================================
print("=" * 70)
print("Step 1: Creating Grid World Environment")
print("=" * 70)
print()

class GridWorld:
    def __init__(self, size=4):
        # size: Grid dimensions (size × size)
        # Example: size=4 creates 4×4 grid (16 states total)
        self.size = size
        
        # Initialize environment
        # reset(): Sets starting position
        self.reset()
        
        # Goal position: Bottom-right corner
        # (size-1, size-1): Last row, last column (0-indexed)
        # Example: size=4 → goal = (3, 3)
        self.goal = (size-1, size-1)
    
    def reset(self):
        """Reset to start state"""
        # Start at top-left corner: (0, 0)
        # This is the initial state for each episode
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """Take action, return (next_state, reward, done)"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        # Get current position
        row, col = self.state
        
        # Update position based on action
        # Ensure we don't go outside grid boundaries
        if action == 0:  # Up
            # max(0, row - 1): Can't go above row 0 (top boundary)
            row = max(0, row - 1)
        elif action == 1:  # Right
            # min(self.size - 1, col + 1): Can't go beyond right boundary
            # self.size - 1 is the last column index (0-indexed)
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            # min(self.size - 1, row + 1): Can't go below bottom boundary
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            # max(0, col - 1): Can't go left of column 0 (left boundary)
            col = max(0, col - 1)
        
        # Update current state to new position
        self.state = (row, col)
        
        # Calculate reward and check if episode is done
        # Reward: -1 for each step (encourages finding shortest path)
        # +10 for reaching goal (large positive reward)
        if self.state == self.goal:
            reward = 10   # Large positive reward for reaching goal
            done = True    # Episode finished
        else:
            reward = -1   # Small negative reward for each step (encourages efficiency)
            done = False  # Episode continues
        
        # Return: (new_state, reward, done_flag)
        # This is the standard RL environment interface
        return self.state, reward, done
    
    def state_to_index(self, state):
        """Convert (row, col) to index"""
        # Convert 2D position to 1D index for Q-table
        # Formula: index = row * grid_width + col
        # Example: (2, 3) in 4×4 grid → 2*4 + 3 = 11
        return state[0] * self.size + state[1]

# ============================================================================
# Step 2: Q-Learning Agent
# ============================================================================
print("=" * 70)
print("Step 2: Implementing Q-Learning Agent")
print("=" * 70)
print()

class QLearningAgent:
    def __init__(self, num_states=16, num_actions=4, lr=0.1, gamma=0.9, epsilon=0.1):
        # num_states: Total number of states in environment
        # For 4×4 grid: 16 states (4×4 = 16)
        self.num_states = num_states
        
        # num_actions: Number of possible actions
        # For grid world: 4 actions (up, right, down, left)
        self.num_actions = num_actions
        
        # lr (learning rate): How much to update Q-values
        # lr=0.1: Update Q-value by 10% towards target
        # Higher lr = faster learning but might overshoot
        self.lr = lr  # Learning rate
        
        # gamma (discount factor): How much we value future rewards
        # gamma=0.9: Value future rewards at 90% of immediate rewards
        # Higher gamma = care more about long-term rewards
        # Range: 0.0 (only immediate) to 1.0 (infinite horizon)
        self.gamma = gamma  # Discount factor
        
        # epsilon (exploration rate): Probability of random action
        # epsilon=0.1: 10% chance of random action (exploration)
        # 90% chance of best action (exploitation)
        # Balance: Explore to learn vs exploit what we know
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: Stores Q(state, action) values
        # Shape: (num_states, num_actions)
        # Q-value = Expected future reward for taking action in state
        # Initialize to zeros (no knowledge initially)
        # Example: Q[0, 1] = Q-value for state 0, action 1
        self.q_table = np.zeros((num_states, num_actions))
    
    def state_to_index(self, state, env_size=4):
        """Convert state to index"""
        # Convert 2D position (row, col) to 1D index
        # Formula: index = row * grid_width + col
        # Example: (2, 3) in 4×4 grid → 2*4 + 3 = 11
        return state[0] * env_size + state[1]
    
    def choose_action(self, state, env_size=4, training=True):
        """Epsilon-greedy action selection"""
        # Convert state to index for Q-table lookup
        state_idx = self.state_to_index(state, env_size)
        
        # Epsilon-greedy policy: balance exploration vs exploitation
        # With probability epsilon: explore (try random action)
        # With probability (1-epsilon): exploit (use best known action)
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            # np.random.random(): Returns random float in [0, 1)
            # If < epsilon, choose random action to explore
            # np.random.randint(self.num_actions): Returns random integer in [0, num_actions)
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: best action (highest Q-value)
            # np.argmax(): Returns index of maximum value
            # self.q_table[state_idx]: Gets Q-values for all actions in this state
            # Returns action with highest Q-value (best known action)
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done, env_size=4):
        """Update Q-value using Bellman equation"""
        # Convert states to indices for Q-table access
        state_idx = self.state_to_index(state, env_size)
        next_state_idx = self.state_to_index(next_state, env_size)
        
        # Current Q-value: Q(state, action)
        # self.q_table[state_idx, action]: Gets Q-value for this state-action pair
        current_q = self.q_table[state_idx, action]
        
        # Maximum Q-value for next state
        # If episode is done (reached goal), there's no next state
        if done:
            next_max_q = 0  # No future reward (episode ended)
        else:
            # Get maximum Q-value over all possible actions in next state
            # This represents best possible future reward from next state
            # np.max(): Finds maximum value in array
            next_max_q = np.max(self.q_table[next_state_idx])
        
        # Bellman equation: Q(s,a) = Q(s,a) + lr × [reward + gamma × max(Q(s',a')) - Q(s,a)]
        # Target Q-value: reward + gamma × max(Q(next_state, all_actions))
        # gamma (discount factor): How much we value future rewards (0.9 = 90%)
        # Higher gamma = care more about long-term rewards
        target_q = reward + self.gamma * next_max_q
        
        # Update Q-value: move current Q-value towards target
        # Learning rate (lr) controls how much to update
        # Higher lr = faster learning, but might overshoot
        # Formula: new_Q = old_Q + lr × (target - old_Q)
        new_q = current_q + self.lr * (target_q - current_q)
        
        # Store updated Q-value in Q-table
        self.q_table[state_idx, action] = new_q

# ============================================================================
# Step 3: Training
# ============================================================================
print("=" * 70)
print("Step 3: Training Q-Learning Agent")
print("=" * 70)
print()

def train_agent(agent, env, episodes=1000):
    """Train Q-learning agent"""
    # Lists to track performance over episodes
    episode_rewards = []  # Total reward per episode
    episode_steps = []    # Number of steps per episode
    
    # Train for specified number of episodes
    for episode in range(episodes):
        # Reset environment to start state
        # env.reset(): Returns initial state (usually (0, 0))
        state = env.reset()
        
        # Track episode statistics
        total_reward = 0  # Cumulative reward for this episode
        steps = 0         # Number of steps taken
        done = False      # Whether episode is finished
        
        # Episode loop: continue until goal reached or episode ends
        while not done:
            # ===== ACTION SELECTION =====
            # Choose action using epsilon-greedy policy
            # agent.choose_action(): Returns action (0-3)
            # training=True: Enable exploration (random actions with probability epsilon)
            action = agent.choose_action(state, env.size, training=True)
            
            # ===== TAKE ACTION =====
            # Execute action in environment
            # env.step(action): Returns (next_state, reward, done)
            #   next_state: New position after action
            #   reward: Immediate reward (-1 for step, +10 for goal)
            #   done: True if goal reached, False otherwise
            next_state, reward, done = env.step(action)
            
            # ===== UPDATE Q-VALUE =====
            # Learn from experience using Bellman equation
            # agent.update(): Updates Q(state, action) based on reward and next state
            # This is the core of Q-learning: learning from experience
            agent.update(state, action, reward, next_state, done, env.size)
            
            # Move to next state for next iteration
            state = next_state
            
            # Accumulate statistics
            total_reward += reward  # Add reward to total
            steps += 1              # Increment step counter
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            # Calculate average over last 100 episodes
            # episode_rewards[-100:]: Last 100 rewards
            # np.mean(): Average value
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode+1}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}")
    
    # Return training statistics for analysis
    return episode_rewards, episode_steps

# Train agent
env = GridWorld(size=4)
agent = QLearningAgent(num_states=16, num_actions=4, lr=0.1, gamma=0.9, epsilon=0.1)

rewards, steps = train_agent(agent, env, episodes=1000)
print()

# ============================================================================
# Step 4: Evaluation
# ============================================================================
print("=" * 70)
print("Step 4: Evaluating Trained Agent")
print("=" * 70)
print()

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent"""
    # Set epsilon to 0: No exploration, only exploitation
    # agent.epsilon = 0: Always choose best action (no random actions)
    # This tests how well agent performs using learned policy
    agent.epsilon = 0  # No exploration
    
    # List to store rewards from each evaluation episode
    total_rewards = []
    
    # Run multiple evaluation episodes
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Run episode until goal reached
        while not done:
            # Choose action (no exploration, only best action)
            # training=False: Disable exploration (epsilon already 0, but explicit)
            action = agent.choose_action(state, env.size, training=False)
            
            # Take action
            state, reward, done = env.step(action)
            total_reward += reward
        
        # Store reward for this episode
        total_rewards.append(total_reward)
    
    # Calculate statistics
    # np.mean(): Average reward across episodes
    # np.std(): Standard deviation (measures consistency)
    # Lower std = more consistent performance
    return np.mean(total_rewards), np.std(total_rewards)

mean_reward, std_reward = evaluate_agent(agent, env)
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
print()

# ============================================================================
# Step 5: Visualize Results
# ============================================================================
print("=" * 70)
print("Step 5: Visualizing Results")
print("=" * 70)
print()

# Training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards, alpha=0.6, label='Episode Reward')
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), 
         label='Moving Average (100)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress: Rewards', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(steps, alpha=0.6, label='Steps per Episode')
plt.plot(np.convolve(steps, np.ones(100)/100, mode='valid'), 
         label='Moving Average (100)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Training Progress: Steps', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_training_progress.png', dpi=150, bbox_inches='tight')
print("Saved: rl_training_progress.png")
print()

# Q-table visualization
def visualize_q_table(agent, env_size=4):
    """Visualize learned Q-values"""
    q_table_reshaped = agent.q_table.reshape(env_size, env_size, 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        im = ax.imshow(q_table_reshaped[:, :, action], cmap='hot')
        ax.set_title(f'Q-values for {action_names[action]}', fontweight='bold')
        ax.set_xticks(range(env_size))
        ax.set_yticks(range(env_size))
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('q_table_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: q_table_visualization.png")
    print()

visualize_q_table(agent)

# Print learned policy
print("=" * 70)
print("Learned Policy (Best Action per State)")
print("=" * 70)
print()

action_symbols = ['↑', '→', '↓', '←']
policy = np.argmax(agent.q_table, axis=1).reshape(4, 4)

for i in range(4):
    row_str = ""
    for j in range(4):
        if (i, j) == (3, 3):
            row_str += " G "
        else:
            row_str += f" {action_symbols[policy[i, j]]} "
    print(row_str)

print()
print("=" * 70)
print("Project 11 Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Created Grid World environment")
print("  ✅ Implemented Q-learning algorithm")
print("  ✅ Trained agent to navigate")
print("  ✅ Visualized Q-values and policy")
print("  ✅ Agent learned optimal path!")
print()
