# Project 11: Reinforcement Learning

> **Build AI agents that learn through trial and error using Q-learning**

**Difficulty**: ⭐⭐⭐⭐ Expert  
**Time**: 5-6 hours  
**Prerequisites**: Steps 0-6 (Python, NumPy, basic ML concepts)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Build Learning Agent](#problem-build-learning-agent)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you reinforcement learning - how AI learns through interaction. You'll build:

1. **Grid World Environment**: Simple navigation task
2. **Q-Learning Agent**: Learns optimal policy
3. **Policy Evaluation**: Measure agent performance
4. **Advanced Techniques**: Epsilon-greedy, value iteration

### Why Reinforcement Learning?

- **Different paradigm**: Learn from rewards, not labels
- **Real-world applications**: Games, robotics, autonomous systems
- **Cutting-edge**: Powers AlphaGo, self-driving cars
- **Foundation**: Prepares for advanced RL

---

## 📋 Problem: Build Learning Agent

### Task

Build an agent that learns to navigate a grid world:
1. **Create Environment**: Grid with obstacles and goal
2. **Implement Q-Learning**: Learn action values
3. **Train Agent**: Learn optimal policy
4. **Evaluate Performance**: Measure learning progress

### Learning Objectives

- Understand RL concepts (agent, environment, reward)
- Implement Q-learning algorithm
- Use epsilon-greedy exploration
- Evaluate agent performance

---

## 🧠 Key Concepts

### 1. RL Components

**Agent**: Makes decisions (the AI)
**Environment**: The world agent interacts with
**State**: Current situation
**Action**: What agent does
**Reward**: Feedback from environment
**Policy**: Strategy for choosing actions

### 2. Q-Learning

**Q-value**: Q(state, action) = Expected future reward

**Bellman Equation**:
```
Q(s,a) = Q(s,a) + lr × [reward + γ × max(Q(s',a')) - Q(s,a)]
```

**Process**: Update Q-values based on rewards

### 3. Exploration vs Exploitation

**Exploration**: Try new actions (learn)
**Exploitation**: Use best known actions (perform)
**Epsilon-greedy**: Balance both

---

## 🚀 Step-by-Step Guide

### Step 1: Create Environment

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.reset()
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2)]  # Example obstacles
    
    def reset(self):
        """Reset to start state"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """Take action, return (next_state, reward, done)"""
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
        
        # Rewards
        if self.state == self.goal:
            reward = 10
            done = True
        elif self.state in self.obstacles:
            reward = -5
            done = False
        else:
            reward = -1  # Small penalty for each step
            done = False
        
        return self.state, reward, done
    
    def state_to_index(self, state):
        """Convert (row, col) to index"""
        return state[0] * self.size + state[1]
```

### Step 2: Q-Learning Agent

```python
class QLearningAgent:
    def __init__(self, num_states=16, num_actions=4, lr=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: Q(state, action)
        self.q_table = np.zeros((num_states, num_actions))
    
    def state_to_index(self, state, env_size=4):
        """Convert state to index"""
        return state[0] * env_size + state[1]
    
    def choose_action(self, state, env_size=4, training=True):
        """Epsilon-greedy action selection"""
        state_idx = self.state_to_index(state, env_size)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: best action
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done, env_size=4):
        """Update Q-value using Bellman equation"""
        state_idx = self.state_to_index(state, env_size)
        next_state_idx = self.state_to_index(next_state, env_size)
        
        # Current Q-value
        current_q = self.q_table[state_idx, action]
        
        # Maximum Q-value for next state
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_idx])
        
        # Bellman equation
        target_q = reward + self.gamma * next_max_q
        new_q = current_q + self.lr * (target_q - current_q)
        self.q_table[state_idx, action] = new_q
```

### Step 3: Training

```python
def train_agent(agent, env, episodes=1000):
    """Train Q-learning agent"""
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.choose_action(state, env.size, training=True)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Update Q-value
            agent.update(state, action, reward, next_state, done, env.size)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode+1}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}")
    
    return episode_rewards, episode_steps

# Train agent
env = GridWorld(size=4)
agent = QLearningAgent(num_states=16, num_actions=4, lr=0.1, gamma=0.9, epsilon=0.1)

rewards, steps = train_agent(agent, env, episodes=1000)
```

### Step 4: Evaluation

```python
def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent"""
    agent.epsilon = 0  # No exploration
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, env.size, training=False)
            state, reward, done = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

# Evaluate
mean_reward, std_reward = evaluate_agent(agent, env)
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

### Step 5: Visualize Q-Table

```python
def visualize_q_table(agent, env_size=4):
    """Visualize learned Q-values"""
    q_table_reshaped = agent.q_table.reshape(env_size, env_size, 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        im = ax.imshow(q_table_reshaped[:, :, action], cmap='hot')
        ax.set_title(f'Q-values for {action_names[action]}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

visualize_q_table(agent)
```

---

## 📊 Expected Results

### Training Progress

```
Episode 100: Avg Reward=-15.23, Avg Steps=15.23
Episode 200: Avg Reward=-8.45, Avg Steps=8.45
Episode 500: Avg Reward=-6.12, Avg Steps=6.12
Episode 1000: Avg Reward=-6.00, Avg Steps=6.00
```

### Evaluation

```
Mean Reward: 4.00 ± 0.00
(Agent consistently reaches goal in 6 steps)
```

### Q-Table

- Higher Q-values near goal
- Clear policy learned
- Optimal path identified

---

## 💡 Extension Ideas

1. **Deep Q-Networks (DQN)**
   - Use neural networks instead of Q-table
   - Handle large state spaces
   - Experience replay

2. **Policy Gradients**
   - Learn policy directly
   - REINFORCE algorithm
   - Actor-Critic methods

3. **Advanced Environments**
   - CartPole
   - Mountain Car
   - Atari games

---

## ✅ Success Criteria

- ✅ Build working environment
- ✅ Implement Q-learning correctly
- ✅ Agent learns optimal policy
- ✅ Visualize Q-values
- ✅ Understand RL concepts

---

**Ready to build learning agents? Let's train with rewards!** 🚀
