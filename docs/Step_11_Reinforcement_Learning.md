# Step 11: Reinforcement Learning Basics

> **Learn how AI learns through trial and error in interactive environments**

**Time**: ~90 minutes  
**Prerequisites**: Steps 0-6 (Neural networks, PyTorch basics)

---

## 🎯 Learning Objectives

By the end of this step, you'll understand:
- What reinforcement learning is and how it differs from supervised learning
- Key RL concepts: agent, environment, state, action, reward, policy
- Q-learning algorithm and how it works
- Policy gradients (conceptual)
- Deep Q-Networks (DQN)
- Real-world RL applications
- Challenges in reinforcement learning

---

## 📚 What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties.

### Key Components

- **Agent**: The AI that makes decisions
- **Environment**: The world the agent interacts with
- **State**: Current situation/observation
- **Action**: What the agent does
- **Reward**: Feedback from environment (positive or negative)
- **Policy**: Strategy for choosing actions (π: state → action)

### The Learning Process

```
1. Agent observes current state
2. Agent chooses action based on policy
3. Environment transitions to new state
4. Agent receives reward
5. Agent updates policy to maximize future rewards
6. Repeat
```

---

## 🔄 RL vs Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Data** | Labeled examples | Experience (interactions) |
| **Teacher** | Provides correct answers | Only provides rewards |
| **Dataset** | Static | Dynamic (generated during learning) |
| **Feedback** | Immediate (correct/incorrect) | Delayed (rewards) |
| **Goal** | Minimize error on dataset | Maximize cumulative reward |

### When to Use RL

✅ **Use RL when**:
- No labeled data available
- Learning through interaction is possible
- Sequential decision making
- Long-term planning needed

❌ **Don't use RL when**:
- Labeled data is available
- Simple classification/regression
- Immediate feedback available
- Static dataset

---

## 🎮 Simple Environment: Grid World

### Grid World Setup

- **4×4 grid**: Agent navigates from start to goal
- **Start**: Top-left corner (0, 0)
- **Goal**: Bottom-right corner (3, 3)
- **Actions**: Up, Right, Down, Left
- **Rewards**: -1 per step, +10 for reaching goal

### Why Grid World?

- **Simple**: Easy to understand and visualize
- **Discrete**: Finite states and actions
- **Perfect for learning**: Demonstrates RL concepts clearly

---

## 🧠 Q-Learning: Learning Action Values

### What is Q-Learning?

**Q-Learning** is a value-based RL algorithm that learns the quality (Q-value) of taking each action in each state.

**Q-value**: `Q(state, action) = Expected future reward`

### Q-Table

A table storing Q-values for each (state, action) pair:

```
State | Action Up | Action Right | Action Down | Action Left
------|-----------|--------------|-------------|------------
(0,0) |    -5.2   |     2.1      |    -3.4     |    -5.8
(0,1) |    -4.1   |     5.3      |    -2.2     |    -6.1
...
```

### Bellman Equation

The core of Q-learning updates:

```
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate
- **r**: Immediate reward
- **γ (gamma)**: Discount factor (how much we value future rewards)
- **s'**: Next state
- **max(Q(s',a'))**: Best Q-value in next state

### Exploration vs Exploitation

**Exploration**: Try new actions to discover better strategies  
**Exploitation**: Use best known actions to maximize reward

**Epsilon-Greedy Policy**:
- With probability ε: Choose random action (exploration)
- Otherwise: Choose best action (exploitation)

**Balance**: Start with high ε (explore), decrease over time (exploit)

---

## 📈 Policy Gradients (Conceptual)

### What are Policy Gradients?

**Policy Gradients** learn the policy directly instead of learning Q-values.

### Key Differences

| Aspect | Q-Learning | Policy Gradients |
|--------|-----------|------------------|
| **What's learned** | Q-values | Policy |
| **Policy type** | Deterministic (best action) | Stochastic (probability distribution) |
| **Actions** | Discrete | Continuous or discrete |
| **Update** | Value-based | Gradient-based |

### REINFORCE Algorithm

1. **Collect trajectory**: (s₀, a₀, r₀, s₁, a₁, r₁, ...)
2. **Calculate returns**: Discounted cumulative rewards
3. **Update policy**: Increase probability of actions with high returns

### When to Use

- **Continuous actions** (e.g., robot control)
- **Stochastic policies** needed
- **High-dimensional action spaces**

---

## 🎯 Deep Q-Networks (DQN)

### What is DQN?

**Deep Q-Networks (DQN)** combine Q-learning with deep neural networks to handle complex state spaces.

### Why DQN?

- **Large state spaces**: Can't use Q-table
- **Continuous states**: Need function approximation
- **Raw inputs**: Can learn from pixels/images

### Key Innovations

#### 1. Experience Replay

- **Store experiences** in a buffer: (state, action, reward, next_state)
- **Sample random batches** for training
- **Breaks correlation** between consecutive samples
- **More stable training**

#### 2. Target Network

- **Separate network** for computing Q-targets
- **Updated less frequently** (every N steps)
- **Stabilizes training** by keeping targets fixed

#### 3. Neural Network Architecture

```
Input: State (can be image pixels)
  ↓
Hidden Layers: Feature extraction
  ↓
Output: Q-values for each action
```

### DQN Algorithm

```
1. Initialize Q-network and target network
2. For each episode:
   a. Observe state
   b. Choose action (epsilon-greedy)
   c. Take action, observe reward and next state
   d. Store experience in replay buffer
   e. Sample batch from replay buffer
   f. Update Q-network using target network
   g. Periodically update target network
```

---

## 🌍 RL Applications

### Games

- **AlphaGo**: Beat world champion at Go
- **OpenAI Five**: Dota 2 team
- **Atari Games**: Learn from pixels
- **Chess/Checkers**: Self-play learning

### Robotics

- **Robot Control**: Manipulation, navigation
- **Autonomous Navigation**: Path planning
- **Grasping**: Learning to pick up objects

### Autonomous Vehicles

- **Path Planning**: Optimal routes
- **Decision Making**: Lane changes, merging
- **Traffic Optimization**: Reduce congestion

### Business

- **Recommendation Systems**: Personalized suggestions
- **Resource Allocation**: Optimal distribution
- **Trading Algorithms**: Stock trading
- **Dynamic Pricing**: Adjust prices based on demand

### Healthcare

- **Treatment Optimization**: Personalized treatments
- **Drug Discovery**: Find new compounds
- **Medical Diagnosis**: Decision support

---

## ⚠️ Challenges in RL

### 1. Sample Efficiency

**Problem**: Requires many interactions with environment

**Impact**: Can be slow to learn, expensive

**Solutions**:
- Better exploration strategies
- Transfer learning
- Imitation learning

### 2. Exploration vs Exploitation

**Problem**: Balance trying new actions vs using known good ones

**Impact**:
- Too much exploration: Slow learning
- Too much exploitation: Miss better strategies

**Solutions**:
- Epsilon-greedy with decay
- Upper Confidence Bound (UCB)
- Thompson Sampling

### 3. Credit Assignment

**Problem**: Which actions led to the reward?

**Impact**: Hard to learn from delayed rewards

**Solutions**:
- Discounted rewards
- Eligibility traces
- Temporal difference learning

### 4. Stability

**Problem**: Training can be unstable

**Impact**: Hard to reproduce, sensitive to hyperparameters

**Solutions**:
- Target networks
- Experience replay
- Gradient clipping

---

## 💻 Code Example

### Q-Learning Implementation

```python
class QLearningAgent:
    def __init__(self, num_states, num_actions, lr=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((num_states, num_actions))
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
    
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        next_max_q = 0 if done else np.max(self.q_table[next_state])
        
        # Bellman equation
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q
```

---

## 📊 Visualizations

The step includes:
1. **Grid World Visualization** - See the environment
2. **Learning Curves** - Rewards and steps over time
3. **Q-Table Heatmap** - Learned Q-values
4. **Learned Policy** - Best action for each state

---

## ✅ Key Takeaways

1. **RL learns through interaction** - No labeled data needed
2. **Q-learning learns action values** - Q(s,a) = expected reward
3. **Exploration vs Exploitation** - Balance is crucial
4. **DQN handles complex states** - Neural networks + Q-learning
5. **RL has unique challenges** - Sample efficiency, stability

---

## 🚀 Next Steps

After this step, you can:
- Understand how RL differs from supervised learning
- Implement basic Q-learning
- Understand DQN concepts
- Recognize RL applications
- Know when to use RL

**To dive deeper**:
- Try OpenAI Gym environments
- Implement DQN with PyTorch
- Learn advanced algorithms (PPO, A3C)
- Explore continuous control (DDPG, TD3)

---

## 📚 Additional Resources

- [OpenAI Gym](https://gym.openai.com/) - Standard RL environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Spinning Up](https://spinningup.openai.com/) - Educational RL resource
- [Deep RL Course](https://www.youtube.com/watch?v=zR11FLZ-O9M) - Video lectures

---

## 🎓 Summary

**Reinforcement Learning** is a powerful paradigm where agents learn through interaction:

1. **Q-Learning**: Value-based, learns Q(s,a)
2. **Policy Gradients**: Policy-based, learns π(s)
3. **DQN**: Combines Q-learning with neural networks

**Key insight**: RL doesn't need labeled data - it learns from rewards!

---

**Happy Learning!** 🎮🤖
