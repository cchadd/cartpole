import gym
from models.reinforce import Reinforce


# Load environment and parameters
env_id = 'CartPole-v1'
learning_rate = 0.01
gamma = 1 
seed = 1235

config = {
    'env_id': env_id,
    'learning_rate': learning_rate,
    'seed': seed,
    'gamma': gamma
}

print("Current config is:")
print(config)


# Defines Agent
agent = Reinforce(config)

# Trains it !
agent.train(n_trajectories=50, n_update=60)

# Evaluate the agent !
agent.evaluate()