# Reinforcement Learning: Q-Learning for Gridworld

This project implements a simple **Q-learning** algorithm to solve a Gridworld decision-making problem. The agent learns to navigate a 5x5 grid while avoiding obstacles and reaching the goal state, maximizing its cumulative reward.

## Overview

### Problem Description
- The environment is a 5x5 grid.
- The agent starts at the top-left corner `(0, 0)`.
- The goal state is located at `(4, 4)`.
- Obstacles are present at specific grid locations (e.g., `(2, 2)` and `(3, 3)`).
- The agent earns:
  - **+10** for reaching the goal.
  - **-10** for hitting an obstacle.
  - **-1** for every other step.

### Q-Learning
Q-learning is a **model-free reinforcement learning algorithm** that learns the optimal policy for an agent by updating a Q-table, which stores the expected future rewards for each state-action pair.

## Features
- Implements a simple Gridworld environment.
- Uses the **epsilon-greedy policy** for exploration and exploitation.
- Tracks rewards and learns an optimal policy over multiple episodes.
- Outputs the learned policy after training.


