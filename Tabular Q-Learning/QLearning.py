import gym
import numpy as np
import matplotlib.pyplot as plt

# Define the discount rate
# The reader is encouraged to experiment with this.
GAMMA = 0.9

# Set up the taxi environment
environment = gym.make('Taxi-v3')

# Build the Q-table
NUM_STATES = environment.observation_space.n
NUM_ACTIONS = environment.action_space.n
Qtable = np.zeros((NUM_STATES, NUM_ACTIONS))

scores = []
for episode in range(500):  # Run for 500 games
    # We're starting a new episode, so reset the environment.
    state = environment.reset()
    score = 0

    done = False
    while not done:
        # Pick the action with the highest Q-value
        action = Qtable[state].argmax()
        # Perform that action  --  The syntax here will become clear in the next section.
        next_state, reward, done, _ = environment.step(action)
        # Update the Q table according to the bellman equation
        Qtable[state, action] = reward + GAMMA * Qtable[next_state].max()

        score += reward
        state = next_state

    scores.append(score)

# Build the plot
plt.title('Taxi-v3 performance')
plt.ylabel('score')
plt.xlabel('episode')
plt.plot(scores, '.')
plt.show()