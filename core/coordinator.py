import time
from typing import List, Dict, Any, Optional
from core.agent import Agent

class AgentCoordinator:
    """
    Coordinates the execution of multiple agents.

    This class can manage multiple agents running in parallel (simulated)
    or sequence, distributing tasks or simply running them.
    """

    def __init__(self):
        self.agents: List[Agent] = []
        self.agent_ids: List[str] = []

    def register_agent(self, agent: Agent, agent_id: str):
        """Add an agent to the coordinator."""
        self.agents.append(agent)
        self.agent_ids.append(agent_id)

    def run_step_all(self, observations: List[Any]) -> List[Any]:
        """
        Run a single step for all agents.

        Args:
            observations: List of observations, one for each agent.

        Returns:
            List of actions, one for each agent.
        """
        if len(observations) != len(self.agents):
            raise ValueError(f"Expected {len(self.agents)} observations, got {len(observations)}")

        actions = []
        for agent, obs in zip(self.agents, observations):
            action = agent.act(obs)
            actions.append(action)

        return actions

    def broadcast_update(self, *args, **kwargs):
        """Call update() on all agents."""
        results = []
        for agent in self.agents:
            res = agent.update(*args, **kwargs)
            results.append(res)
        return results

    def run_episodes(self,
                     envs: Any,
                     num_episodes: int = 1,
                     max_steps: int = 100,
                     render: bool = False) -> Dict[str, List[float]]:
        """
        Run episodes for all agents in parallel using a Batched Environment.

        Args:
            envs: An Environment instance with batch_size == num_agents
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
            render: Whether to render

        Returns:
            Dictionary mapping agent_id to list of episode returns
        """
        if envs.batch_size != len(self.agents):
            raise ValueError(f"Env batch size {envs.batch_size} != num agents {len(self.agents)}")

        results = {aid: [] for aid in self.agent_ids}

        for ep in range(num_episodes):
            observations = envs.reset()
            # If batch_size=1, envs.reset() returns single obs, wrap it
            if envs.batch_size == 1:
                observations = [observations]

            episode_returns = [0.0] * len(self.agents)
            dones = [False] * len(self.agents)

            # Reset agents internal state if needed
            for agent in self.agents:
                if hasattr(agent, 'reset'):
                    agent.reset()
                # Initialize hidden states for stateful agents
                # We assume agents handle their own init or we use a helper
                # For this demo, we try to set hidden state if the agent has the method
                if hasattr(agent, 'reset_hidden'):
                    # Create dummy hidden state?
                    # This is tricky without model info.
                    # We'll assume the agent handles it or fails if not provided.
                    # For the example script, we'll use stateless models or handle it there.
                    pass

            for step in range(max_steps):
                if render:
                    envs.render()

                actions = self.run_step_all(observations)

                # Step environment
                next_observations, rewards, new_dones, infos = envs.step(actions)

                # Handle batch_size=1 case
                if envs.batch_size == 1:
                    next_observations = [next_observations]
                    rewards = [rewards]
                    new_dones = [new_dones]
                    infos = [infos]

                # Observe
                for i, agent in enumerate(self.agents):
                    if not dones[i]:
                        agent.observe(
                            observation=next_observations[i],
                            action=actions[i],
                            reward=rewards[i],
                            terminated=new_dones[i], # Simplified done
                            truncated=False,
                            info=infos[i]
                        )
                        episode_returns[i] += rewards[i]
                        dones[i] = new_dones[i]

                observations = next_observations

                if all(dones):
                    break

            for i, aid in enumerate(self.agent_ids):
                results[aid].append(episode_returns[i])

        return results
