import gfootball.env as football_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pickle
import normalizer

parser = argparse.ArgumentParser(description='PyTorch Google Football Q Learning')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--actor_lr', default=0.0003, type=float)
parser.add_argument('--critic_lr', default=0.0003, type=float)
parser.add_argument('--eps', default=0.9, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--q-iteration', default=100, type=int)
parser.add_argument('--augm', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mem_cap', default=6000, type=int)
args = parser.parse_args()
print(args)

#Initialize Constants
MEM_CAP = args.mem_cap #Lab GTX1080Tis have about 8 Gigs on board but we keep some buffer
NUM_S = 115
NUM_A = 19
AUGMS = ["none","no_clipping","no_reward_norm","adam_al","orth_init"]
SEEDS = [1,2,3,4,5,6,7,8,9,10]
CHOSEN_AUGM = AUGMS[args.augm]
CHOSEN_SEED = SEEDS[args.seed]
print(f"We are using augmentation {CHOSEN_AUGM}")
print(f"We are using seed {CHOSEN_SEED}")
print(f"We are using environment 11_vs_11_easy_stochastic")
print(f"We are using algorithm PPO")

class PolicyNet(nn.Module):
    """Actor Critic Policy Evaluators for PPO with special Orthogonal Initialization as mentioned in paper"""

    def __init__(self, num_states =NUM_S, num_actions=NUM_A, hidden_dim=64):
        super().__init__()

        self.shared_w = nn.Sequential(
            self.get_fc(num_states, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor_fc = self.get_fc(hidden_dim, num_actions, g=np.sqrt(2))
        self.critic_fc = self.get_fc(hidden_dim, 1)

    def get_fc(self, in_dim, out_dim, g=1):
        layer = nn.Linear(in_dim, out_dim)
        nn.init.orthogonal_(layer.weight.data, gain=g)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def forward(self, x):
        x = self.shared_w(x)
        pi_a = F.softmax(self.actor_fc(x), dim=1)
        dist = torch.distributions.Categorical(pi_a)
        value_f = self.critic_fc(x)
        return value_f, dist



class ActorCriticNetDDPG(nn.Module):
    """Actor and Critic for DDPG"""

    def __init__(self, num_states =NUM_S, num_actions=NUM_A, hidden_dim=64, td3=False):
        super().__init__()
        self.td3 = td3
        self.actor_local = nn.Sequential(
            self.get_fc(num_states, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, num_actions, g=np.sqrt(2)))

        self.actor_target = nn.Sequential(
            self.get_fc(num_states, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, num_actions, g=np.sqrt(2)))

        self.critic_local = nn.Sequential(
            self.get_fc(num_states + num_actions, hidden_dim), #Evaluates taken actions for a state
            nn.Tanh(),
            self.get_fc(hidden_dim, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, 1))

        self.critic_target = nn.Sequential(
            self.get_fc(num_states + num_actions, hidden_dim),  # Evaluates taken actions for a state
            nn.Tanh(),
            self.get_fc(hidden_dim, hidden_dim),
            nn.Tanh(),
            self.get_fc(hidden_dim, 1))

        self.actors = nn.ModuleList([self.actor_target, self.actor_local])
        if td3:
            self.critic_local_2 = nn.Sequential(
                self.get_fc(num_states + num_actions, hidden_dim),  # Evaluates taken actions for a state
                nn.Tanh(),
                self.get_fc(hidden_dim, hidden_dim),
                nn.Tanh(),
                self.get_fc(hidden_dim, 1))

            self.critic_target_2 = nn.Sequential(
                self.get_fc(num_states + num_actions, hidden_dim),  # Evaluates taken actions for a state
                nn.Tanh(),
                self.get_fc(hidden_dim, hidden_dim),
                nn.Tanh(),
                self.get_fc(hidden_dim, 1))
            self.critics = nn.ModuleList([self.critic_target, self.critic_local, self.critic_target_2])
        else:
            self.critics = nn.ModuleList([self.critic_target, self.critic_local])

    def get_fc(self, in_dim, out_dim, g=1):
        layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def soft_update_target_network(self, tau):

        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def forward_actor(self, x, type='local'):
        if type == 'local':
            actions = self.actor_local(x)
        elif type == 'target':
            actions = self.actor_target(x)
        actions = F.softmax(actions,dim=1)
        return actions

    def forward_critic(self, x, type='local'):
        if type == 'local':
            value_f = self.critic_local(x)
        elif type == 'target':
            value_f = self.critic_target(x)
        return value_f

    def forward_critic_2(self, x, type='local'):
        if type == 'local':
            value_f = self.critic_local_2(x)
        elif type == 'target':
            value_f = self.critic_target_2(x)
        return value_f






class PPO:
    def __init__(self):
        super(PPO, self).__init__()
        self.net = PolicyNet()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.00008)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, NUM_S, eta_min=0, last_epoch=-1) ## added by Lawrence
        self.normalizer = normalizer.RunningMeanStd()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_batch_counter = 0
        self.memory = np.zeros((MEM_CAP, NUM_S + 5))
        self.memory_batch = np.zeros((1, NUM_S * 2 + 4))
        # Keep track of amount in memory to make sure there is no buffer overflow
        # Memory: NEW_STATE, OLD_STATE, ACTION, REWARD, Policy Prob
        self.values = torch.zeros(128).cuda()
        self.selected_prob = torch.zeros(128).cuda() #Batch Size
        #OpenAI Parameters
        if CHOSEN_AUGM == "no_clipping":
            self.epsilon = 0
        else:
            self.epsilon = 0.27
        self.gamma = 0.993
        self.lamb = 0.95
        self.value_l_p = 0.5
        self.entropy_p = 0.01

    def learn(self):
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEM_CAP, args.batch_size)
        batch_memory = self.memory[sample_index, :]

        states = torch.FloatTensor(batch_memory[:, :NUM_S]).cuda()
        advantages = torch.FloatTensor(batch_memory[:, NUM_S:NUM_S + 1]).cuda()
        rewards_to_go = torch.FloatTensor(batch_memory[:, NUM_S + 1:NUM_S + 2]).cuda()
        values = torch.FloatTensor(batch_memory[:, NUM_S + 2: NUM_S + 3]).cuda()
        actions = torch.LongTensor(batch_memory[:, NUM_S + 3: NUM_S + 4]).cuda()
        selected_prob = torch.FloatTensor(batch_memory[:, NUM_S+4:NUM_S+5]).cuda()


        values_new, dist_new = self.net(states)
        values_new = values_new.flatten()
        selected_prob_new = dist_new.log_prob(actions)

        # Compute the PPO loss
        prob_ratio = torch.exp(selected_prob_new) / torch.exp(selected_prob)

        a = prob_ratio * advantages
        b = torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        ppo_loss = -1 * torch.mean(torch.min(a, b))

        value_pred_clipped = values + (values_new - values).clamp(-self.epsilon, self.epsilon)
        value_losses = (values_new - rewards_to_go) ** 2
        value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        value_loss = value_loss.mean()
        entropy_loss = torch.mean(dist_new.entropy())

        loss = ppo_loss + self.value_l_p * value_loss - self.entropy_p * entropy_loss

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), .5)
        self.optim.step()
        return float(loss)

    def choose_action(self, s, shape_env_a):
        s = torch.unsqueeze(torch.FloatTensor(s), 0).cuda()

        step_values, dist = self.net(s)

        #Get actions from policy distribution
        action = dist.sample()
        policy_probs = dist.log_prob(action)

        return action.cpu().tolist()[0], policy_probs ,step_values

    def store_transition(self, s, a, r, s_plus_one, policy_probs, done, v):
        # Store in memory, PPO also needs an advantage to be stored in memory

        #Make a batch of not done and calculate advantages and normalized rewards when done
        transition = np.hstack((s, [a, r], s_plus_one, policy_probs.cpu().detach(), [v.squeeze(0).cpu().detach()]))
        self.memory_batch = np.vstack((self.memory_batch, transition))

        # Handle normalization and Advantages
        if done == 1:
            values = self.memory_batch[1:, NUM_S + 3: NUM_S + 4]
            rewards = self.memory_batch[1:, NUM_S + 1:NUM_S + 2]
            rewards = self.reward_norm(rewards)
            advantages = self.gae(rewards, values)
            rewards_to_go = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            actions = self.memory_batch[1:, NUM_S:NUM_S + 1].astype(int)
            states = self.memory_batch[1:, :NUM_S]
            policy_probs = self.memory_batch[1:, NUM_S + 3: NUM_S + 4]

            ppo_needed_data = np.hstack((states,advantages, rewards_to_go, values, actions,  policy_probs))
            index = self.memory_counter % MEM_CAP
            if index + advantages.shape[0] <= MEM_CAP:
                self.memory[index: index + advantages.shape[0], :] = ppo_needed_data
            else:
                self.memory[index:MEM_CAP, :] = ppo_needed_data[:MEM_CAP-index, :]

            self.memory_batch = np.zeros((1, NUM_S * 2 + 4))
            self.memory_counter += advantages.shape[0]



    def reward_norm(self, data, update_data=None, center=True,
                         clip_limit=10):

        """Reward Normalization"""
        if update_data is not None:
            # Update the statistics with different data than we're normalizing
            self.normalizer.update(update_data.reshape((-1,) + self.normalizer.shape))
        else:
            self.normalizer.update(data.reshape((-1,) + self.normalizer.shape))
        if center:
            data = data - self.normalizer.mean

        if not CHOSEN_AUGM == "no_reward_norm":
            data = data / np.sqrt(self.normalizer.var + 1e-8)

        data = np.clip(data, -clip_limit, clip_limit)

        return data

    def gae(self, rewards, values):
        """Generalized advantage estimate."""
        N = rewards.shape[0]  #Batch
        T = rewards.shape[1]
        gae_step = np.zeros((N,))
        advantages = np.zeros((N, T))
        for t in reversed(range(T - 1)):
            one_step_td_error = rewards[:, t] + self.gamma * values[:, t + 1] - values[:, t]
            current_gae_step = one_step_td_error + self.gamma * self.lamb * gae_step
            advantages[:, t] = current_gae_step
        return advantages





env = football_env.create_environment(
    env_name="11_vs_11_easy_stochastic",
    representation='simple115',
    number_of_left_players_agent_controls=1,
    stacked=False, logdir='/tmp/football',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False)


ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

max_episodes = 100000000 #Arbitrarily large number
i_episode = 0
state = env.reset()
steps = 0
double = False
dueling =False
model = PPO()
model.net.cuda()
sum_sample_number = 0
loss_sum = 0
train_dict = {"Episodes": [], "Loss": [], "Reward": [], "Step": []}
test_dict = {"Episodes": [], "Reward": []}
loss_results = []
total_done = False
ep_reward_max = 0
while i_episode < max_episodes:
    
    if CHOSEN_AUGM == "adam_al":
        
        scheduler.step()
    
    i_episode += 1
    env.reset()
    steps = 0
    done = False
    ep_reward = 0
    sample_number = 0
    loss_sum = 0
    while not done: #PPO also needs advantages computed here
        sample_number += 1
        sum_sample_number += 1
        #	print(reward)
        action = model.choose_action(state, ENV_A_SHAPE)
        next_state, reward, done, infor = env.step(action)
        model.store_transition(state, action, reward, next_state)
        ep_reward += reward
        if model.memory_counter >= MEM_CAP:
            loss_sum += model.learn()
            if done:
                print(
                    f"Episode: {i_episode} Sample: {sample_number}, Episode Reward is {ep_reward}, Avg. Loss {loss_sum / sample_number}")
                train_dict["Episodes"].append(i_episode)
                train_dict["Loss"].append(loss_sum)
                train_dict["Reward"].append(ep_reward)
                train_dict["Step"].append(sample_number)


        if done:
            break

    if i_episode % 100 is 0:
        print("Testing 10 Episodes...")
        ep_reward = 0
        for i in range(10):
            env.reset()
            steps = 0
            done = False
            while not done:
                action, policy_probs, step_values = model.choose_action(state, ENV_A_SHAPE)
                next_state, reward, done, infor = env.step(action)
                ep_reward += reward
                if done:
                    break

        print(f"[Test] Episode: {i_episode}  Avg. Episode Reward is {ep_reward / 10}")
        test_dict["Episodes"].append(i_episode)
        test_dict["Reward"].append(ep_reward / 10)
        if (ep_reward / 10 > ep_reward_max):
            print(f"New Best Avg Reward: {ep_reward / 10}")
            print("Saving Training Results...")
            pickle_train = open(f"PPO_11_vs_11_simple_stochastic_train.p", "wb")
            pickle.dump(train_dict, pickle_train)
            pickle_test = open(f"PPO_11_vs_11_simple_stochastic_test.p", "wb")
            pickle.dump(test_dict, pickle_test)
            ep_reward_max = (ep_reward / 10)

        if (ep_reward / 10 == 1.):
            total_done = True

    state = next_state
    if total_done:
        print("Succesfully Trained!")
        break
