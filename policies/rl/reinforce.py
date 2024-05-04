import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        # Dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Defines policy architecture
        self.seq_model = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
      return torch.softmax(self.seq_model(x).flatten(), dim=-1)

# Initialize environment and optimizer. Feel free to change!
env = gym.make('CartPole-v1')
input_size = 4
output_size = env.action_space.n
hidden_size = 64
policy_net = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_net.parameters(), lr=0.005)

avg_rewards = []
std_rewards = []

# Training hyperparameters. Feel free to change!
num_episodes = 1500
gamma = 0.99

# Training loop
for episode in range(num_episodes):
    episode_rewards = []
    log_probs = []

    state,_ = env.reset()
    done = False
    trunc = False

    while not done and not trunc:
      # Sample action from the policy network. For simplicity we only use single trajectory
      # to update. But feel free to change this into batch learning!
      state_tensor = torch.from_numpy(state).float().unsqueeze(0)
      action_probs = policy_net(state_tensor)
      action_dist = Categorical(action_probs)
      action = action_dist.sample()
      log_prob = action_dist.log_prob(action)
      log_probs.append(log_prob)

      # Take action and observe next state and reward
      next_state, reward, done, trunc, _ = env.step(action.item())
      episode_rewards.append(reward)
      state = next_state


    # Now use the episode_rewards list, the log_prob list to construct a loss function, on which you can
    # backward propagate and optimize with optimizer. First, try to compute the discounted cumulative reward
    # for every step in the trajectory.
    discounted_reward = []
    for h in range(len(episode_rewards)):
      h_discounted_reward = 0
      for t in range(len(episode_rewards) - h):
        h_discounted_reward += gamma**t * episode_rewards[t + h]
      discounted_reward.append(h_discounted_reward)
    discounted_rewards = torch.tensor(discounted_reward, dtype=torch.float32)

    # Now compute the policy loss with discounted_rewards and log_prob. Use this loss to run policy gradient.
    log_probs = torch.stack(log_probs)
    optimizer.zero_grad()
    loss = -1 * torch.sum(log_probs * discounted_rewards)
    loss.backward()
    optimizer.step()

    # Record the cumulative reward and its deviation once every 100 episodes
    if (episode + 1) % 100 == 0:
      avg_reward, std_reward = evaluate_neural(env, policy_net)
      print(f'Episode [{episode + 1}/{num_episodes}], Cumulative Reward: {avg_reward}')
      avg_rewards.append(avg_reward)
      std_rewards.append(std_reward)