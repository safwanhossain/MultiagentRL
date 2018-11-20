#!/usr/bin/python3
from utils.base_model import BaseModel
from actor import Actor
from critic import Critic
from utils.buffer import Buffer
from environment.particle_env import make_env, visualize
import torch
import time

""" Run the maac training regime. There is a global critic for
each agent with an attention mechanism. For both the critic and actor loss, they use an entropy
term to encourage exploration. They also use a baseline to reduce variance."""

class MAAC(BaseModel):
    def __init__(self, env, use_gpu=True):
        """
        Initialize all aspects of the model
        :param env: Environment model will be used in
        """
        super(MAAC, self).__init__(use_gpu=use_gpu)
        self.env = env
        self.batch_size = 1024
        self.num_entries_per_update = self.batch_size
        self.seq_len = 100
        self.discount = 0.95
        self.n_agents = env.n
        self.action_size = env.action_size
        self.obs_size = env.agent_obs_size
        self.use_gpu = use_gpu
        self.sequence_length = 25
        self.epochs = 10
        self.lr_critic = 0.01
        self.lr_actor = 0.01
        self.tau = 0.002
        self.alpha = 0.2
        self.attend_tau = 0.04
        self.steps_per_update = 100
        self.num_updates = 4
        # The buffer to hold all the information of an episode
        self.buffer = Buffer(1000000, 100, self.batch_size, self.n_agents,
                             env.agent_obs_size, env.global_state_size, env.action_size)
        
        # maac does NOT do parameter sharing - each agent has it's own network and optimizer
        self.agents = [Actor(self.obs_size, self.action_size, self.device) for i in range(self.n_agents)]
        self.target_agents = [Actor(self.obs_size, self.action_size, self.device) for i in range(self.n_agents)]
        self.agent_optimizers = [torch.optim.Adam(agent.get_params(), lr=self.lr_actor) for agent in self.agents]
        
        # We usually have two versions of the critic, as TD lambda is trained thru bootstraping. self.critic
        # is the "True" critic and target critic is the one used for training
        self.critic = Critic(observation_size=self.obs_size,action_size=self.action_size, device=self.device,
                             num_agents=self.n_agents, embedding_dim=128, attention_heads=4)
        self.target_critic = Critic(observation_size=self.obs_size,action_size=self.action_size, device=self.device,
                                    num_agents=self.n_agents,  embedding_dim=128, attention_heads=4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=1e-3)

        self.set_params()

    def format_buffer_data(self, curr_agent_obs, next_agent_obs, curr_state, next_state, actions, rewards):
        """
        Reshape buffer data to correct dimensions
        """
        curr_agent_obs = curr_agent_obs.view(self.n_agents, self.batch_size, self.obs_size)
        next_agent_obs = next_agent_obs.view(self.n_agents, self.batch_size, self.obs_size)
        actions = actions.view(self.n_agents, self.batch_size, self.action_size)
        rewards = rewards.view(self.batch_size, 1)
        ret = curr_agent_obs, next_agent_obs, actions, rewards
        return [data.to(self.device) for data in ret]

    def update_critic(self):
        """ train the critic usinng batches of observatios and actions. Training is based on the 
        TD lambda method with a target critic"""
        assert(not self.buffer.is_empty())
        batch_size = self.batch_size

        # collect the batch
        curr_agent_obs_batch, next_agent_obs_batch, action_batch, reward_batch = \
            self.format_buffer_data(*self.buffer.sample_from_buffer(ordered=False, full_episode=False))
        
        # get the critic values for all agents
        critic_values = self.critic(curr_agent_obs_batch, action_batch)
        
        # Sample a batch of actions given the next observation (for the target network)
        # TODO: This can be sped up if we don't use target agents
        agent_probs = torch.zeros(self.n_agents, batch_size, 1).to(self.device)
        next_joint_actions = torch.zeros(self.n_agents, batch_size, self.action_size).to(self.device)
        for n in range(self.n_agents):
            probs = self.target_agents[n](next_agent_obs_batch[n], 0)
            action_ids = torch.multinomial(probs, 1)
            next_joint_actions[n] = torch.FloatTensor(probs.shape).to(self.device).fill_(0).scatter_(1, action_ids, 1)
            agent_probs[n] = probs.gather(1, action_ids)

        # get the target critic values for all agents
        target_values = self.target_critic(next_agent_obs_batch, next_joint_actions)

        # compute the critic loss
        critic_loss = 0
        mse_loss = torch.nn.MSELoss().to(self.device)
        for n in range(self.n_agents):
            y_i = reward_batch + self.discount*(target_values[n] - self.alpha*torch.log(agent_probs[n]))
            critic_loss += mse_loss(critic_values[n], y_i.detach())
        
        # backpropagate the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return torch.mean(critic_loss).item() / self.seq_len

    
    def update_actor(self):
        """ Train the actor using the Q function to approximate long term reward, use a baseline to
        reduce variance, and use an entropy term to encourage exploration"""
        assert(not self.buffer.is_empty())
        batch_size = self.batch_size

        # collect the batch
        curr_agent_obs_batch, next_agent_obs_batch, action_batch, reward_batch = \
            self.format_buffer_data(*self.buffer.sample_from_buffer(ordered=False, full_episode=False))

        # Use the current observations to sample a set of actions (for the target network)
        agent_probs = torch.zeros(self.n_agents, batch_size, 1).to(self.device)
        next_joint_actions = torch.zeros(self.n_agents, batch_size, self.action_size).to(self.device)
        for n in range(self.n_agents):
            probs = self.agents[n](curr_agent_obs_batch[n], eps=0)
            action_ids = torch.multinomial(probs, 1)
            next_joint_actions[n] = torch.FloatTensor(probs.shape).to(self.device).fill_(0).scatter_(1, action_ids, 1)
            agent_probs[n] = probs.gather(1, action_ids)
        
        # Get the Q values for each agent at the current observation
        all_action_q, curr_action_q = self.critic(curr_agent_obs_batch, action_batch, ret_all_actions=True)

        sum_loss = 0.
        # compute the loss using baseline and entropy term
        for n in range(self.n_agents):
            log_pi = torch.log(agent_probs[n]).to(self.device)
            baseline = (all_action_q[n].to(self.device)*agent_probs[n].to(self.device)).sum(dim=1, keepdim=True)
            target = curr_action_q[n].to(self.device) - baseline
            target = (target - target.mean()) / target.std()    # make it 0 mean and 1 var (idk why??)
            loss = (log_pi*(self.alpha*log_pi - target)).mean()

            self.agent_optimizers[n].zero_grad()
            loss.backward(retain_graph=True)
            self.agent_optimizers[n].step()
            sum_loss += torch.mean(loss).item()

        return sum_loss / (self.n_agents * self.seq_len)

    def update_target_networks(self):
        """ The target networks should basically mimic the trained networks with some 1-tau probability"""
        # First update the non-attention modules for critic
        for target_param, param in zip(self.target_critic.get_non_attention_parameters(),\
                self.critic.get_non_attention_parameters()):
            target_param.data.copy_(target_param.data * (self.tau) + param.data * (1 - self.tau))
        
        # Then update the attention modules for critic
        for target_param, param in zip(self.target_critic.get_attention_parameters(),\
                self.critic.get_attention_parameters()):
            target_param.data.copy_(target_param.data * self.attend_tau + param.data * (1 - self.attend_tau))
        
        # Now update the target agents
        for n in range(self.n_agents):
            for target_param, param in zip(self.target_agents[n].get_params(), self.agents[n].get_params()):
                target_param.data.copy_(target_param.data * self.tau + param.data * (1 - self.attend_tau))

    def policy(self, obs, prev_action, n, eps):
        """
        return probability distribution over actions given observation
        :param obs: observation from environment for agent n
        :param prev_action: last action of this agent
        :param n: agent index
        :return: probability distribution of type torch tensor
        """
        return self.agents[n](obs.view(-1, self.obs_size).to(self.device), eps=eps)[0]

    def update(self, epoch):
        """
        Update model
        :param epoch: current epoch
        :return: critic loss, average agent loss
        """
        for i in range(self.num_updates):
            critic_loss = self.update_critic()
            actor_loss = self.update_actor()
        self.update_target_networks()
        return critic_loss, actor_loss

def main():
    number_of_agents = 2
    env = make_env(number_of_agents)
    model = MAAC(env)

    st = time.time()
    model.train()
    print("Time taken for {0:d} epochs {1:10.4f}".format(model.epochs, time.time() - st))

    visualize(model)

if __name__ == "__main__":
    main()

