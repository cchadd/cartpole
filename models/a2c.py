import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from torch import optim
import torch.nn.functional as F
import gym
from gym.wrappers import Monitor
from models.actor_net import ActorNetwork
from models.values_net import ValueNetwork


class A2CAgent:

    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']
        
        # Our two networks
        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)
        
        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(), lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=config['actor_network']['learning_rate'])
        
    # Hint: use it during training_batch
    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network
        
        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        cum_rewards = []
        advantages = []


        for i in range(len(rewards)):
          
            cum_reward = rewards[i]

            t = 0

            # Sums all the discounted rewards over a trajectory
            while not dones[i + t] and t + i + 2 < len(rewards):
                
                cum_reward += rewards[i + t + 1] * self.gamma ** (t + 1)
                t += 1
            
            # Checks if this is the last trajectory for bootstrapping
            if dones[i:].sum()==0:
                cum_rewards.append(cum_reward + self.gamma ** t * next_value.data)
        
            else:
                cum_rewards.append(cum_reward)

        return cum_rewards

        

    def training_batch(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        states = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):

                observation = torch.Tensor(observation)

                # Stores s_t
                states[i] = observation

                # Selects an action
                action = self.actor_network.select_action(observation)

                obs, reward, done, _ = self.env.step(int(action))

                # Stores the action
                actions[i] = action

                # Stores corresponding reward
                rewards[i] = reward

                # Stores observations (s_t+1)
                observations[i] = obs

                # Stores status of trajectory
                dones[i] = done

                # Stores the estimates of the value function network
                values[i] =  self.value_network(torch.Tensor(observation))
                
                # step
                observation = obs

                if dones[i]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(torch.Tensor(observation))
            
            
            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns = self._returns_advantages(rewards, dones, values, next_value)

            # Converts to tensors
            returns = torch.Tensor(returns)
            states = torch.Tensor(states)
            rewards = torch.Tensor(rewards)
            
            # Learning step !
            self.optimize_model(observations, actions, returns, rewards, states)

            # Test it every 20 epochs
            if epoch % 20 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(20)]))
                print(f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs -1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()
                    
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
        plt.show()
        
        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, rewards, states):
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)
        actions = F.one_hot(torch.tensor(actions))


        value_function_mod = self.value_network(observations)

        # Lets learn the value function
        for i in range(20):


            # Defines loss for V
            loss_value = F.mse_loss(value_function_mod, returns)

            # Resets gradient to zero
            self.value_network_optimizer.zero_grad()

            # Computes gradient
            loss_value.backward()

            # Update Value function
            self.value_network_optimizer.step()

            value_function_mod = self.value_network(observations)


        delta = []

        # Computes advantages
        for i in range(len(rewards)):
            delta.append(rewards[i] + self.gamma * self.value_network(observations[i]) - self.value_network(states[i]))

        delta = torch.tensor(delta)

       
        # Computes probas & logprobas of corresponding actions
        probas = torch.sum(self.actor_network(states) * actions, axis=1)
        log_probas = torch.sum(torch.log(self.actor_network(states)) * actions, axis=1)

        # Defines loss & entropy
        loss_act = - torch.mean(torch.mul(log_probas, delta))
        entropy = - torch.mean(torch.mul(log_probas, probas))
       
        # Resets gardient to zero
        self.actor_network_optimizer.zero_grad()
        
        # Defines total loss
        loss = loss_act + entropy

        # Computes gradient
        loss.backward()
       
        # Update the policy
        self.actor_network_optimizer.step()

    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            policy = self.actor_network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward
            
        env.close()
        return reward_episode