import torch
from .base import RLAlgorithmBase
from torchkit.networks import FlattenMlp
import os
import torch.nn as nn

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import optax
import torch.nn.functional as F
import torchkit.pytorch_utils as ptu
from torchkit.core import PyTorchModule

class custom_mlp(nn.Module):
    def __init__(self, hidden_sizes, input_size, action_dim):
        super(custom_mlp, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, *inputs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return self.model(flat_inputs)


class DQN_Reimplemented(RLAlgorithmBase):
    name = "dqn_reimplemented"
    continuous_action = False

    def __init__(self, init_eps=1.0, end_eps=0.01, schedule_steps=1000, **kwargs):
        self.epsilon_schedule = optax.linear_schedule(
            init_value=init_eps,
            end_value=end_eps,
            transition_steps=schedule_steps,
        )
        self.count = 0

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        return custom_mlp(input_size=input_size, action_dim=action_dim, hidden_sizes=hidden_sizes)



    def select_action(self, qf, observ, deterministic):
        print("determ", deterministic)
        action_logits = qf(observ)
        num_actions = action_logits.shape[-1]
        batch_size = action_logits.shape[0]

        random_action = torch.randint(num_actions, size=(batch_size,)).to(ptu.device)
        optimal_action = torch.argmax(action_logits, dim=-1)

        eps = self.epsilon_schedule(self.count).item()

        # Create exploration-exploitation mask
        exploration_mask = torch.multinomial(input=ptu.FloatTensor([1 - eps, eps]), num_samples=batch_size, replacement=True)

        # Choose action based on exploration-exploitation mask
        action = exploration_mask * random_action + (1 - exploration_mask) * optimal_action

        self.count += 1
        one_hot_action = F.one_hot(action.long(), num_classes=num_actions).float()

        return one_hot_action



    def critic_loss(
        self,
        markov_critic: bool,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        """Calculates the critic loss for both Markovian and non-Markovian critics.

        Args:
            markov_critic (bool): If True, use a Markovian critic approach.
            critic, critic_target: Critic and target critic models.
            observs (Tensor): Current observations.
            actions (Tensor): Actions taken.
            rewards (Tensor): Received rewards.
            dones (Tensor): Done flags.
            next_observs (Tensor, optional): Next observations for Markovian critic.

        Returns:
            tuple: Predicted and target Q values.
        """
        # Calculate next value predictions with no gradients
        with torch.no_grad():
            quality = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=None,
            )  # (T+1, B, A)
            target_quality = critic_target(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=None,
            )  # (T+1, B, A)



            # Calculate target Q values
            next_actions = torch.argmax(quality, dim=-1, keepdim=True)
            next_target_q = target_quality.gather(dim=-1, index=next_actions)
            q_target = rewards + (1.0 - dones) * gamma * next_target_q
            q_target = q_target[1:] 

        # Calculate predicted Q values
        q_pred = self.calculate_predicted_values(critic, observs, actions, rewards)

        return q_pred, q_target


    def calculate_predicted_values(self, critic, observs, actions, rewards):
        v_pred = critic(prev_actions=actions[:-1], rewards=rewards[:-1], observs=observs[:-1], current_actions=None,)
        stored_actions = torch.argmax(actions[1:], dim=-1, keepdim=True)
        q_pred = v_pred.gather(dim=-1, index=stored_actions)
        return q_pred
    
