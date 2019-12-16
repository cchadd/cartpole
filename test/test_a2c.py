from models.a2c import A2CAgent
import warnings
warnings.filterwarnings('ignore')

# Defines environment and parameters
env_id = 'CartPole-v1'
value_learning_rate = 0.001
actor_learning_rate = 0.001
gamma = 0.99
entropy = 1
seed = 1

config_a2c = {
    'env_id': env_id,
    'gamma': gamma,
    'seed': seed,
    'value_network': {'learning_rate': value_learning_rate},
    'actor_network': {'learning_rate': actor_learning_rate},
    'entropy': entropy
}

print("Current config_a2c is:")
print(config_a2c)

# Creates agent
agent = A2CAgent(config_a2c)

# Trains it !
agent.training_batch(1000, 256)

# Evaluate Agent !
agent.evaluate(render=True)