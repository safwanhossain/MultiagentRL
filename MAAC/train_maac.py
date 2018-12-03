#!/usr/bin/python3

from pdb import set_trace as bp
from comet_ml import Experiment
import torch
import numpy as np
import torch.nn as nn
import utils
from critic import Global_Critic
from actor import Actor_Policy
from episode_buffer import Buffer
import marl_env
from datetime import datetime

""" Run the MAAC training regime. Here we have E parallel environments, and a global critic for
each agent with an attention mechanism. For both the critic and actor loss, they use an entropy
term to encourage exploration. They also use a baseline to reduce variance."""

# In MAAC, we need to run a number of parallel environments. Set them up here
def make_parallel_environments(env_name, num_environments):
    # create multiple environments each with a different seed
    parallel_envs = []
    for i in range(num_environments):
        env = marl_env.make_env('simple_spread')
        np.random.seed(i*1000)
        env.seed(i*1000)
        parallel_envs.append(env)
    return parallel_envs


class MAAC():
    def __init__(self, parallel_envs, n_agents, action_size, agent_obs_size, log):
        """
        :param batch_size:
        :param seq_len: horizon
        :param discount: discounting factor
        :param n_agents:
        :param action_size: size of the discrete action space, actions are as one-hot vectors
        :param obs_size:
        :param state_size: size of global state (position and velocity of each agent + location of landmarks)
        """
        
        super(MAAC, self).__init__()
        self.parallel_envs = parallel_envs
        self.batch_size = 1020
        self.seq_len = 100
        self.gamma = 0.95 
        self.n_agents = n_agents
        self.action_size = action_size
        self.obs_size = agent_obs_size
        self.gpu_mode = True
        self.sequence_length = 25
        self.episodes = 10000
        self.critic_lr = 0.0005
        self.agent_lr = 0.0001
        self.tau = 0.04
        self.alpha = 0.1
        self.attend_tau = 0.002
        self.steps_per_update = 100
        self.num_updates = 4
        self.buffer_clear_mod = 750

        # for logging metrics
        self.experiment = log
        self.agent_step = 0
        self.log_step = 0

        # The buffer to hold all the information of an episode
        self.buffer = Buffer(self.n_agents, self.obs_size, self.action_size, len(self.parallel_envs)) 
               
        # MAAC does NOT do parameter sharing - each agent has it's own network and optimizer
        self.agents = [Actor_Policy(input_size=self.obs_size, action_size=self.action_size) for i in range(self.n_agents)]
        self.target_agents = [Actor_Policy(input_size=self.obs_size, action_size=self.action_size) for i in range(self.n_agents)]
        self.agent_optimizers = [torch.optim.Adam(agent.get_params(), lr=self.agent_lr) for agent in self.agents]
        
        # We usually have two versions of the critic, as TD lambda is trained thru bootstraping. self.critic
        # is the "True" critic and target critic is the one used for training
        self.critic = Global_Critic(observation_size=self.obs_size, \
                action_size=self.action_size, num_agents=self.n_agents, attention_heads=4)
        self.target_critic = Global_Critic(observation_size=self.obs_size, \
                action_size=self.action_size, num_agents=self.n_agents, attention_heads=4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr) 
        
        # get the critic values for all agents
        if self.gpu_mode:
            for agent in self.agents:
                agent.get_network().cuda()
            for agent in self.target_agents:
                agent.get_network().cuda()
            self.critic.cuda()
            self.target_critic.cuda()

    def update_critic(self, batch_size):
        """ train the critic usinng batches of observatios and actions. Training is based on the 
        TD lambda method with a target critic"""
        assert(not self.buffer.is_empty())

        # collect the batch
        curr_agent_obs_batch, next_agent_obs_batch, \
                action_batch, reward_batch = self.buffer.sample_from_buffer(batch_size)
        
        # get the critic values for all agents
        critic_values, reg = self.critic(curr_agent_obs_batch.cuda(), action_batch.cuda(), regularize=True)
        
        # Sample a batch of actions given the next observation (for the target network)
        agent_probs = torch.zeros(self.n_agents, batch_size, 1).cuda()
        next_joint_actions = torch.zeros(self.n_agents, batch_size, self.action_size).cuda()
        for n in range(self.n_agents):
            probs = self.target_agents[n].action(next_agent_obs_batch[n].cuda())
            action_ids = torch.multinomial(probs, 1)
            next_joint_actions[n] = torch.FloatTensor(*probs.shape).cuda().fill_(0).scatter_(\
                    1, action_ids, 1)
            agent_probs[n] = probs.gather(1, action_ids)

        # get the target critic values for all agents
        target_values = self.target_critic(next_agent_obs_batch.cuda(), next_joint_actions.cuda())[0].detach()
        # compute the critic loss
        critic_loss = 0
        mse_loss = torch.nn.MSELoss().cuda()
        for n in range(self.n_agents):
            y_i = reward_batch.cuda() + self.gamma*(target_values[n].cuda() - self.alpha*torch.log(agent_probs[n].cuda()))
            critic_loss += mse_loss(critic_values[n].cuda(), y_i.detach())
            critic_loss += reg[n].cuda()       # attention regularization (not mentioned in paper but implemented in their Github)

        # backpropagate the loss
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        return float(critic_loss.cpu()[0])
    
    def update_agent(self, batch_size):
        """ Train the actor using the Q function to approximate long term reward, use a baseline to
        reduce variance, and use an entropy term to encourage exploration"""
        assert(not self.buffer.is_empty())
        
        # collect the batch
        curr_agent_obs_batch, next_agent_obs_batch, \
                action_batch, reward_batch = self.buffer.sample_from_buffer(batch_size)

        # Use the current observations to sample a set of actions (for the target network)
        agent_probs = torch.zeros(self.n_agents, batch_size, 1).cuda()
        next_joint_actions = torch.zeros(self.n_agents, batch_size, self.action_size).cuda()
        for n in range(self.n_agents):
            probs = self.agents[n].action(curr_agent_obs_batch[n].cuda())
            action_ids = torch.multinomial(probs, 1)
            next_joint_actions[n] = torch.FloatTensor(*probs.shape).cuda().fill_(0).scatter_(\
                    1, action_ids, 1)
            agent_probs[n] = probs.gather(1, action_ids)
        
        # Get the Q values for each agent at the current observation
        all_action_q, curr_action_q = self.critic(curr_agent_obs_batch.cuda(), action_batch.cuda(), \
                ret_all_actions=True)

        # compute the loss using baseline and entropy term
        agent_losses = []
        for n in range(self.n_agents):
            log_pi = torch.log(agent_probs[n]).cuda()
            # Compute baseline as an explicit expectation - pg 5 in paper
            baseline = (all_action_q[n].cuda()*agent_probs[n].cuda()).sum(dim=1, keepdim=True)
            target = curr_action_q[n].cuda() - baseline
            target = (target - target.mean()) / target.std()    # make it 0 mean and 1 var (not mentioned in paper but in github code)
            loss = (log_pi*(self.alpha*log_pi - target)).mean()
            
            self.experiment.log_metric("Q value " + str(n), curr_action_q[n].mean().cpu(), self.agent_step)

            # Disable grad on critic - don't really need it cuz it gets zeroed in critic_update, but be safe
            utils.disable_grad(self.critic)
            loss.backward(retain_graph=True)
            utils.enable_grad(self.critic)
            
            self.agent_optimizers[n].step()
            self.agent_optimizers[n].zero_grad()
            agent_losses.append(float(loss.cpu()[0]))
        self.agent_step += 1
        return agent_losses
    
    def update_target_networks(self):
        """ The target networks should basically mimic the trained networks with some 1-tau probability"""
        # First update the non-attention modules for critic
        for target_param, param in zip(self.target_critic.get_non_attention_parameters(),\
                self.critic.get_non_attention_parameters()):
            target_param.data.copy_(target_param.data * (self.tau) + param.data * (1 - self.tau))
        
        # Then update the attention modules for critic
        for target_param, param in zip(self.target_critic.get_attention_parameters(),\
                self.critic.get_attention_parameters()):
            target_param.data.copy_(target_param.data * (self.attend_tau) + param.data * (1 - self.attend_tau))
        
        # Now update the target agents
        for n in range(self.n_agents):
            for target_param, param in zip(self.target_agents[n].get_params(), self.agents[n].get_params()):
                target_param.data.copy_(target_param.data * (self.tau) + param.data * (1 - self.tau))


    def prep_training(self):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.get_network().train()
        for t_a in self.target_agents:
            t_a.get_network().train()
    
    def prep_rollouts(self):
        for a in self.agents:
            a.get_network().eval()
        for t_a in self.target_agents:
            t_a.get_network().eval()
        
    def train(self):
        num_steps = 0
        self.prep_rollouts()
        for ep in range(0, self.episodes, len(self.parallel_envs)):
            print("Episodes %i-%i of %i" % (ep + 1, ep + 1 + len(self.parallel_envs), self.episodes))
            
            curr_obs_tensor = torch.zeros((len(self.parallel_envs), self.n_agents, self.obs_size))
            for i, env in enumerate(self.parallel_envs):
                obs = env.reset()
                for j in range(self.n_agents):
                    curr_obs_tensor[i,j,:] = torch.FloatTensor(obs[j])

            critic_losses = []
            actor_losses = [[] for i in range(self.n_agents)]
            reward_arr = []
            for s in range(self.seq_len):
                # Get the next action given the observation
                joint_actions = torch.zeros(len(self.parallel_envs), self.n_agents, self.action_size)
                for k, agent in enumerate(self.agents):
                    obs_agent = curr_obs_tensor.permute(1,0,2)[k]
                    dist = agent.action(obs_agent.cuda())    # the environments can be treated as a batch
                    
                    # sample action from pi, convert to one-hot vector
                    action_idx = (torch.multinomial(dist.cpu(), num_samples=1)).numpy().flatten()
                    for l in range(len(self.parallel_envs)):
                        joint_actions[l][k][action_idx[l]] = 1

                next_obs_tensor = torch.zeros(len(self.parallel_envs), self.n_agents, self.obs_size)
                for e, env in enumerate(self.parallel_envs):
                    # get observations, by executing current joint action and store in buffer
                    next_obs_n, reward_n, done_n, info_n = env.step(joint_actions[e])
                    for j in range(self.n_agents):
                        next_obs_tensor[e, j, :] = torch.FloatTensor(next_obs_n[j])

                    reward = reward_n[0]
                    reward_arr.append(reward)
                    self.buffer.add_to_buffer(curr_obs_tensor[e], next_obs_n, joint_actions[e], reward, e)
                    
                # next observation becomes the current ones
                curr_obs_tensor = next_obs_tensor
                num_steps += len(self.parallel_envs)

                critic_loss = []
                agent_loss = [[] for _ in range(self.n_agents)]
                if (self.buffer.length() > self.batch_size and \
                        (num_steps % self.steps_per_update) < len(self.parallel_envs)):
                    self.prep_training()
                    for _ in range(self.num_updates):
                        critic_loss.append(self.update_critic(self.batch_size))
                        agent_losses = self.update_agent(self.batch_size)
                        #for j in range(self.n_agents):
                        #    agent_loss[j].append(agent_losses[j])

                    self.update_target_networks()
                    critic_loss_mean = sum(critic_loss)/len(critic_loss)
                    self.experiment.log_metric("Critic loss", critic_loss_mean, self.log_step)
                    #for j in range(self.n_agents):
                    #    agent_loss_mean = sum(agent_loss[j])/len(agent_loss[j])
                    #    self.experiment.log_metric("Agent " + str(j) + " loss", agent_loss_mean, self.log_step)
                    self.log_step += 1
                    self.prep_rollouts()

            # print the reward at end of each episode
            mean_reward = sum(reward_arr)/len(reward_arr)
            print('Reward', mean_reward)
            self.experiment.log_metric("Reward", mean_reward, ep)

def main():
    parallel_envs = make_parallel_environments("simple_spread", 12)
    
    experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI', project_name="MAAC")
    big_MAAC = MAAC(parallel_envs, n_agents=3, action_size=5, agent_obs_size=18, log=experiment)    
    big_MAAC.train()


if __name__ == "__main__":
    main()

