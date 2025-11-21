"""
Example script demonstrating Multi-Agent Coordination with the new Agent Abstractions.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.coordinator import AgentCoordinator
from meta_rl.maml_agent import MetaMAMLAgent
from adaptation.adaptation_agent import AdaptationAgent
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment

# Simple MLP Model for testing (Fake SSM compatible)
class SimplePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, hidden_state=None):
        # Accept hidden_state to match Adapter/SSM interface
        # Return output, next_hidden_state
        out = self.net(x)
        # Return dummy hidden state if none provided, or pass through
        if hidden_state is None:
            hidden_state = torch.zeros(x.size(0), 1).to(x.device)
        return out, hidden_state

def main():
    print("Initializing Multi-Agent System...")

    # 1. Setup Environment (Batch of 2 for 2 agents)
    env_name = "CartPole-v1" # Continuous state, Discrete action
    num_agents = 2
    print(f"Creating {num_agents} parallel {env_name} environments...")
    env = Environment(env_name=env_name, batch_size=num_agents)

    obs_dim = env.observation_space.shape[0]
    # CartPole has Discrete(2) action space, but our simple policy outputs raw logits/continuous
    # We'll handle action selection inside the loop or assume environment handles float->int if needed
    # Actually Environment wrapper just passes actions to gym.
    # For CartPole, we usually output 2 values (logits) and take argmax, or output 1 value (prob).
    # Let's make a model that outputs 2 values.
    act_dim = 2

    # 2. Create Agents

    # Agent 1: Meta-Learning Agent (MAML)
    print("Creating MetaMAMLAgent...")
    maml_model = SimplePolicy(obs_dim, act_dim)
    maml_algo = MetaMAML(maml_model, inner_lr=0.01)
    agent_maml = MetaMAMLAgent(maml_algo)

    # Agent 2: Test-Time Adaptation Agent
    print("Creating AdaptationAgent...")
    adapt_model = SimplePolicy(obs_dim, act_dim)
    adapt_config = AdaptationConfig(learning_rate=0.01, num_steps=1)
    adapter = Adapter(adapt_model, adapt_config)
    agent_adapter = AdaptationAgent(adapter)

    # Initialize hidden states for agents (required since we made SimplePolicy stateful)
    # Our SimplePolicy expects hidden_state of shape (Batch, 1)
    # Each agent acts on a single environment instance in this loop (batch=1 inside act)
    dummy_hidden = torch.zeros(1, 1)
    agent_maml.reset_hidden(dummy_hidden)
    agent_adapter.reset_hidden(dummy_hidden)

    # 3. Coordinator
    print("Registering agents with Coordinator...")
    coordinator = AgentCoordinator()
    coordinator.register_agent(agent_maml, "MAML_Agent")
    coordinator.register_agent(agent_adapter, "Adapt_Agent")

    # 4. Run Loop
    print("\nRunning 5 episodes...")

    # Custom loop to handle action processing (argmax for discrete)
    num_episodes = 5

    for ep in range(num_episodes):
        obs_list = env.reset()
        episode_returns = [0.0, 0.0]
        done_flags = [False, False]

        # Reset hidden states at start of episode
        agent_maml.reset_hidden(dummy_hidden)
        agent_adapter.reset_hidden(dummy_hidden)

        step_count = 0
        while not all(done_flags) and step_count < 200:
            # Get actions from agents
            # Note: agents return raw tensor output from model (logits)
            raw_actions = coordinator.run_step_all(obs_list)

            # Process actions for CartPole (argmax)
            # We must do this because our SimplePolicy outputs [logit0, logit1]
            processed_actions = []
            for raw_act in raw_actions:
                # raw_act is numpy array
                action = np.argmax(raw_act)
                processed_actions.append(action)

            # Step environment
            next_obs_list, rewards, dones, infos = env.step(processed_actions)

            # Update stats
            for i in range(num_agents):
                if not done_flags[i]:
                    episode_returns[i] += rewards[i]
                    if dones[i]:
                        done_flags[i] = True

                    # Observe (store data)
                    # Here we could trigger adaptation if we wanted
                    coordinator.agents[i].observe(
                        next_obs_list[i], processed_actions[i], rewards[i], dones[i], False, infos[i]
                    )

            obs_list = next_obs_list
            step_count += 1

        print(f"Episode {ep+1}: MAML_Agent Return={episode_returns[0]}, Adapt_Agent Return={episode_returns[1]}")

        # Demonstrate "Update/Adapt" call (Mock)
        # In a real scenario, we would construct support sets here
        print("  Triggering mock adaptation/update...")

        # Create dummy support set
        dummy_x = torch.randn(5, obs_dim)
        dummy_y = torch.randn(5, act_dim) # Targets/Actions

        # MAML Agent Update (Inner Loop)
        # We must provide initial_hidden_state because SimplePolicy is now stateful
        # Batch size of dummy_x is 5
        adapt_hidden = torch.zeros(5, 1)
        res_maml = agent_maml.adapt(dummy_x, dummy_y, initial_hidden_state=adapt_hidden)
        print(f"  MAML Agent Adapt: {res_maml}")

        # Adapter Agent Update
        # Note: Adapter expects 'y' to be target for loss.
        # For RL, this might be next state or value. Here just dummy.
        res_adapt = agent_adapter.adapt(dummy_x, dummy_y)
        print(f"  Adapt Agent Adapt: {res_adapt}")

    print("\nDemo Completed.")

if __name__ == "__main__":
    main()
