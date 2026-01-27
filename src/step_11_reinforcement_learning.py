"""
Step 11 — Reinforcement Learning Basics
Goal: Learn the fundamentals of reinforcement learning and how AI learns through interaction
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Step 11: Reinforcement Learning Basics")
print("=" * 70)
print()
print("Goal: Understand how AI learns through trial and error")
print()

# ============================================================================
# 11.1 What is Reinforcement Learning?
# ============================================================================
print("=== 11.1 What is Reinforcement Learning? ===")
print()
print("Reinforcement Learning (RL) is learning through interaction:")
print("  • Agent takes actions in an environment")
print("  • Environment provides rewards/penalties")
print("  • Agent learns to maximize cumulative reward")
print()
print("Key Components:")
print("  • Agent: The AI that makes decisions")
print("  • Environment: The world the agent interacts with")
print("  • State: Current situation")
print("  • Action: What the agent does")
print("  • Reward: Feedback from environment")
print("  • Policy: Strategy for choosing actions")
print()

# ============================================================================
# 11.2 RL vs Supervised Learning
# ============================================================================
print("=== 11.2 RL vs Supervised Learning ===")
print()
print("Supervised Learning:")
print("  • Learn from labeled examples")
print("  • Teacher provides correct answers")
print("  • Static dataset")
print()
print("Reinforcement Learning:")
print("  • Learn from experience")
print("  • No teacher, only rewards")
print("  • Dynamic environment")
print("  • Learn through trial and error")
print()

# ============================================================================
# 11.3 Simple Environment: Grid World
# ============================================================================
print("=== 11.3 Simple Environment: Grid World ===")
print()

class GridWorld:
    """Simple 4x4 grid world environment"""
    def __init__(self):
        self.size = 4
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.state = (0, 0)  # Start at top-left
        self.goal = (3, 3)  # Goal at bottom-right
        return self.state
    
    def step(self, action):
        """Take action and return (next_state, reward, done)"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        row, col = self.state
        
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
        
        self.state = (row, col)
        
        # Reward: -1 for each step, +10 for reaching goal
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done
    
    def visualize(self):
        """Visualize current state"""
        grid = np.zeros((self.size, self.size))
        grid[self.state[0], self.state[1]] = 1  # Agent
        grid[self.goal[0], self.goal[1]] = 2  # Goal
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid, cmap='RdYlGn', alpha=0.7)
        
        # Add text
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.state:
                    ax.text(j, i, 'A', ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='black')
                elif (i, j) == self.goal:
                    ax.text(j, i, 'G', ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='black')
        
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_title('Grid World Environment', fontsize=14, fontweight='bold')
        ax.grid(True, color='black', linewidth=2)
        plt.tight_layout()
        plt.show()

# Create and visualize environment
env = GridWorld()
print("Grid World Environment:")
print("  • Agent (A) starts at (0, 0)")
print("  • Goal (G) is at (3, 3)")
print("  • Actions: Up, Right, Down, Left")
print("  • Reward: -1 per step, +10 for reaching goal")
print()
env.visualize()

# ============================================================================
# 11.4 Q-Learning: Learning Action Values
# ============================================================================
print("=== 11.4 Q-Learning: Learning Action Values ===")
print()
print("Q-Learning learns the value of taking each action in each state:")
print("  Q(state, action) = Expected future reward")
print()
print("Key Concepts:")
print("  • Q-table: Stores Q-values for each (state, action) pair")
print("  • Exploration vs Exploitation: Try new actions vs use best known")
print("  • Bellman Equation: Update Q-values based on rewards")
print()

class QLearningAgent:
    """Simple Q-learning agent"""
    def __init__(self, num_states=16, num_actions=4, lr=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
    
    def state_to_index(self, state):
        """Convert (row, col) to state index"""
        row, col = state
        return row * 4 + col
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        state_idx = self.state_to_index(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: best action
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Bellman equation"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx, action]
        
        # Maximum Q-value for next state
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_idx])
        
        # Bellman equation: Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_idx, action] = new_q

# Train Q-learning agent
print("Training Q-learning agent...")
agent = QLearningAgent(lr=0.1, gamma=0.9, epsilon=0.1)

episodes = 500
rewards_per_episode = []
steps_per_episode = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = agent.choose_action(state, training=True)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_steps = np.mean(steps_per_episode[-100:])
        print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.1f}")

print()
print("Training complete!")
print()

# Visualize learning progress
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rewards over time
axes[0].plot(rewards_per_episode, alpha=0.6, linewidth=1, color='steelblue')
# Moving average
window = 50
if len(rewards_per_episode) >= window:
    moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(rewards_per_episode)), moving_avg, 
                linewidth=2, color='red', label=f'Moving Average ({window})')
axes[0].set_xlabel('Episode', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Reward', fontsize=12, fontweight='bold')
axes[0].set_title('Q-Learning: Rewards Over Time', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Steps over time
axes[1].plot(steps_per_episode, alpha=0.6, linewidth=1, color='coral')
if len(steps_per_episode) >= window:
    moving_avg_steps = np.convolve(steps_per_episode, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(steps_per_episode)), moving_avg_steps,
                linewidth=2, color='red', label=f'Moving Average ({window})')
axes[1].set_xlabel('Episode', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Steps to Goal', fontsize=12, fontweight='bold')
axes[1].set_title('Q-Learning: Steps to Goal', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize learned Q-table
print("=== Learned Q-Table ===")
print()
print("Q-values for each state-action pair:")
print("(Higher values = better actions)")
print()

fig, ax = plt.subplots(figsize=(10, 8))
q_heatmap = np.zeros((4, 4, 4))  # 4x4 grid, 4 actions

for i in range(4):
    for j in range(4):
        state_idx = i * 4 + j
        q_heatmap[i, j, :] = agent.q_table[state_idx, :]

# Show best action for each state
best_actions = np.argmax(q_heatmap, axis=2)
action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}

im = ax.imshow(np.max(q_heatmap, axis=2), cmap='viridis', aspect='auto')
for i in range(4):
    for j in range(4):
        best_action = best_actions[i, j]
        ax.text(j, i, action_symbols[best_action], ha='center', va='center',
               fontsize=20, fontweight='bold', color='white')
        ax.text(j, i-0.3, f'{np.max(q_heatmap[i, j]):.1f}', ha='center', va='center',
               fontsize=9, color='white')

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xlabel('Column', fontsize=12, fontweight='bold')
ax.set_ylabel('Row', fontsize=12, fontweight='bold')
ax.set_title('Learned Policy (Best Action per State)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Max Q-Value')
plt.tight_layout()
plt.show()

# ============================================================================
# 11.5 Test the Learned Policy
# ============================================================================
print("=== 11.5 Testing Learned Policy ===")
print()

# Test with no exploration
agent.epsilon = 0.0  # No exploration, only exploitation
state = env.reset()
path = [state]
total_reward = 0
steps = 0

print("Testing learned policy (no exploration):")
print(f"Start: {state}")

while True:
    action = agent.choose_action(state, training=False)
    next_state, reward, done = env.step(action)
    path.append(next_state)
    total_reward += reward
    steps += 1
    
    action_names = ['Up', 'Right', 'Down', 'Left']
    print(f"  Step {steps}: Action={action_names[action]}, State={next_state}, Reward={reward}")
    
    state = next_state
    if done:
        break

print()
print(f"Reached goal in {steps} steps!")
print(f"Total reward: {total_reward}")
print(f"Path: {path}")
print()

# ============================================================================
# 11.6 Policy Gradients (Conceptual)
# ============================================================================
print("=== 11.6 Policy Gradients (Conceptual) ===")
print()
print("Policy Gradients learn the policy directly:")
print("  • Instead of learning Q-values, learn policy π(s) → a")
print("  • Policy is a probability distribution over actions")
print("  • Update policy to increase probability of good actions")
print()
print("Key Differences from Q-Learning:")
print("  Q-Learning:")
print("    • Value-based: Learn Q(s,a)")
print("    • Deterministic policy: Choose best action")
print("    • Works well for discrete actions")
print()
print("  Policy Gradients:")
print("    • Policy-based: Learn π(s)")
print("    • Stochastic policy: Sample from distribution")
print("    • Works for continuous actions")
print()
print("REINFORCE Algorithm:")
print("  1. Collect trajectory (s0, a0, r0, s1, a1, r1, ...)")
print("  2. Calculate returns (discounted rewards)")
print("  3. Update policy: Increase probability of actions with high returns")
print()

# ============================================================================
# 11.7 Deep Q-Networks (DQN)
# ============================================================================
print("=== 11.7 Deep Q-Networks (DQN) ===")
print()
print("DQN combines Q-learning with deep neural networks:")
print("  • Use neural network to approximate Q(s,a)")
print("  • Handles large/continuous state spaces")
print("  • Enables learning from raw pixels")
print()
print("Key Innovations:")
print("  1. Experience Replay:")
print("     • Store past experiences in buffer")
print("     • Sample random batches for training")
print("     • Breaks correlation between consecutive samples")
print()
print("  2. Target Network:")
print("     • Separate network for computing targets")
print("     • Updated less frequently")
print("     • Stabilizes training")
print()
print("  3. Neural Network Architecture:")
print("     • Input: State (can be images)")
print("     • Output: Q-values for each action")
print()

# Simple DQN example (conceptual)
print("Example DQN Architecture:")
print("  Input Layer: State representation")
print("  Hidden Layers: Feature extraction")
print("  Output Layer: Q-values for each action")
print()

# ============================================================================
# 11.8 RL Applications
# ============================================================================
print("=== 11.8 RL Applications ===")
print()
print("Reinforcement Learning powers many real-world applications:")
print()
print("🎮 Games:")
print("  • AlphaGo: Beat world champion at Go")
print("  • OpenAI Five: Dota 2 team")
print("  • Atari games: Learn from pixels")
print()
print("🤖 Robotics:")
print("  • Robot control")
print("  • Autonomous navigation")
print("  • Manipulation tasks")
print()
print("🚗 Autonomous Vehicles:")
print("  • Path planning")
print("  • Decision making")
print("  • Traffic optimization")
print()
print("💼 Business:")
print("  • Recommendation systems")
print("  • Resource allocation")
print("  • Trading algorithms")
print()
print("🏥 Healthcare:")
print("  • Treatment optimization")
print("  • Drug discovery")
print("  • Personalized medicine")
print()

# ============================================================================
# 11.9 Challenges in RL
# ============================================================================
print("=== 11.9 Challenges in RL ===")
print()
print("Reinforcement Learning faces unique challenges:")
print()
print("❌ Sample Efficiency:")
print("   • Requires many interactions with environment")
print("   • Can be slow to learn")
print()
print("❌ Exploration vs Exploitation:")
print("   • Balance trying new actions vs using known good ones")
print("   • Too much exploration: Slow learning")
print("   • Too much exploitation: Miss better strategies")
print()
print("❌ Credit Assignment:")
print("   • Which actions led to reward?")
print("   • Delayed rewards make this hard")
print()
print("❌ Stability:")
print("   • Training can be unstable")
print("   • Hyperparameters matter a lot")
print()

# ============================================================================
# 11.10 Key Concepts Summary
# ============================================================================
print("=== 11.10 Key Concepts Summary ===")
print()
print("✅ Reinforcement Learning:")
print("   • Agent learns through interaction")
print("   • Maximizes cumulative reward")
print("   • No labeled data needed")
print()
print("✅ Q-Learning:")
print("   • Value-based method")
print("   • Learns Q(s,a) = expected future reward")
print("   • Uses Bellman equation for updates")
print()
print("✅ Policy Gradients:")
print("   • Policy-based method")
print("   • Learns policy directly")
print("   • Good for continuous actions")
print()
print("✅ Deep Q-Networks:")
print("   • Combines Q-learning with neural networks")
print("   • Handles complex state spaces")
print("   • Uses experience replay and target networks")
print()

# ============================================================================
# 11.11 Next Steps
# ============================================================================
print("=== 11.11 Next Steps ===")
print()
print("To dive deeper into RL:")
print("  1. Try more complex environments (OpenAI Gym)")
print("  2. Implement DQN with PyTorch")
print("  3. Learn about advanced algorithms (PPO, A3C)")
print("  4. Explore continuous control (DDPG, TD3)")
print("  5. Study multi-agent RL")
print()
print("Recommended Resources:")
print("  • OpenAI Gym: Standard RL environments")
print("  • Stable-Baselines3: RL algorithms library")
print("  • Spinning Up: Educational RL resource")
print()

print("=" * 70)
print("Step 11 Complete! You understand reinforcement learning basics!")
print("=" * 70)
