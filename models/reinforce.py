import numpy as np
import torch
from models.baseagent import BaseAgent


class Reinforce(BaseAgent):
    
    def _compute_returns(self, rewards):
        
        cum_rewards = []
        for i in range(len(rewards)):
            cum_reward = 0
            for index, reward in enumerate(rewards[i:]):
                cum_reward += reward * self.gamma ** index
                
            cum_rewards.append(cum_reward)
        
        return cum_rewards
    
        
    def optimize_model(self, n_trajectories):

        
        reward_trajectories = []
        states = []
        actions = []
        rewards = []
        loss = 0

        # samples trajectories
        for traj in range(n_trajectories):

            states_traj = []
            rewards_traj = []
            actions_traj = []
            log_prob_traj = []

            # Initializes the first step
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float)
            done = False
            traj_len = 0

        
            # Enters in the trajectory
            while not done:
                traj_len += 1

                # Selects action
                action = self.model.select_action(state)

                # Computes logproba
                log_prob_traj.append(torch.log(self.model.forward(state)[int(action)]))
                
                # Observes state, reward and done
                state, reward, done, info = self.env.step(int(action))

                # Stores values
                states_traj.append(state)
                actions_traj.append(action)
                rewards_traj.append(reward)

                state = torch.tensor(state, dtype=torch.float)
            
            disc_rew = self._compute_returns(rewards_traj)

            # Converts to array for computation
            logs = np.array([log_prob_traj])
            rew = np.array([disc_rew])

            loss += - np.dot(logs, rew.T)[0, 0]
            reward_trajectories.append(np.sum(rewards_traj))
            
        loss /= n_trajectories

        
        # The following lines take care of the gradient descent step for the variable loss
        # that you need to compute.
        
        # Discard previous gradients
        self.optimizer.zero_grad()

        # Compute the gradient 
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()
        return reward_trajectories