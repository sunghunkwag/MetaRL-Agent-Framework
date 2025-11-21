import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Deque, Union
from collections import deque

from core.agent import Agent
from meta_rl.meta_maml import MetaMAML

class MetaMAMLAgent(Agent):
    """
    Agent wrapper for MetaMAML.

    Handles:
    - Observation storage (buffering)
    - Interaction (act) using functional forward passes
    - Adaptation (inner loop)
    - Meta-updates (outer loop)
    """

    def __init__(self,
                 meta_maml: MetaMAML,
                 device: str = 'cpu',
                 buffer_size: int = 1000):
        self.meta_maml = meta_maml
        self.device = device

        # Internal state for stateful models (e.g. RNNs)
        self.hidden_state = None

        # Fast weights (adapted parameters)
        self.fast_weights = None

        # Buffer to store trajectory for adaptation/update
        # Format: list of (obs, action, reward, next_obs, done)
        self.buffer = []
        self.buffer_size = buffer_size

    def reset(self):
        """Reset agent state (hidden state, buffer, but NOT fast weights unless specified)."""
        self.hidden_state = None
        self.buffer = []

    def act(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Select action using current fast weights (or meta weights if none).
        """
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float().to(self.device)
        else:
            obs_tensor = observation.to(self.device)

        # Add batch dimension if missing
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            if self.meta_maml._stateful:
                # If first step, initialize hidden state
                if self.hidden_state is None:
                    # Assuming standard hidden state init (zeros) is handled by model
                    # or we need to know shape.
                    # For now, passing None to functional_forward usually implies init inside if supported,
                    # BUT MetaMAML.functional_forward raises error if hidden is None for stateful.
                    # We need a way to init hidden state.
                    # Let's assume the model handles None or we catch it.
                    # Looking at MetaMAML code: "if hidden_state is None: raise ValueError"
                    # So we must provide it.
                    # We need to know the hidden shape.
                    # Let's try to infer or require user to pass it via a separate method if needed.
                    # For robustness, let's assume the user must handle state init or we rely on a default.
                    pass

                # NOTE: Handling hidden state init is tricky without knowing model details.
                # We will assume the user sets self.hidden_state manually or the model allows a special call.
                # However, for this wrapper, we'll try to create zeros if we can, or fail gracefully.
                pass

            # Use functional_forward
            # If self.fast_weights is None, it uses meta-parameters
            if self.meta_maml._stateful:
                if self.hidden_state is None:
                     # Try to infer batch size and create dummy hidden state if possible?
                     # Or just rely on the user calling reset_hidden() which we should add?
                     # Let's add a helper.
                     raise RuntimeError("Stateful model requires hidden_state. Call reset_hidden() first.")

                output, next_hidden = self.meta_maml.functional_forward(
                    obs_tensor, self.hidden_state, params=self.fast_weights
                )
                self.hidden_state = next_hidden
            else:
                output = self.meta_maml.functional_forward(
                    obs_tensor, None, params=self.fast_weights
                )

        return output.cpu().numpy().squeeze(0)

    def observe(self, observation: Any, action: Any, reward: float, terminated: bool, truncated: bool, info: Dict) -> None:
        """Store experience in buffer."""
        # We store (observation, action, reward, terminated)
        # Next observation is passed in the next act() call or we assume we just need the current tuple.
        # For MAML usually we need inputs (obs) and targets (action/reward/next_obs).
        # If this is RL, targets might be value targets or actions (behavior cloning).
        # Let's assume standard RL tuple.
        self.buffer.append({
            "obs": observation,
            "action": action,
            "reward": reward,
            "done": terminated or truncated,
            "info": info
        })
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def adapt(self, inputs: torch.Tensor, targets: torch.Tensor, num_steps: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Adapt using provided support set.

        Args:
            inputs: Support set inputs (x)
            targets: Support set targets (y)
            num_steps: Number of gradient steps
            **kwargs: Additional arguments (e.g., initial_hidden_state)
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Prepare initial hidden state for adaptation if needed
        initial_hidden = kwargs.get('initial_hidden_state')

        # If stateful and no hidden state provided, try to default or fail
        if self.meta_maml._stateful and initial_hidden is None:
             # Warning: This might fail if model strictly requires hidden state and we can't infer it.
             # We rely on the user passing it in kwargs for now.
             pass

        try:
            self.fast_weights = self.meta_maml.adapt_task(
                inputs, targets, initial_hidden_state=initial_hidden, num_steps=num_steps
            )
        except ValueError as e:
            if "initial_hidden_state" in str(e):
                 # Fallback for stateful
                 # Create a dummy hidden state (this assumes we know the shape... which we don't easily)
                 # This is a limitation of the generic wrapper without model knowledge.
                 # We will return error.
                 return {"status": "failed", "error": str(e)}
            raise e

        return {"status": "adapted", "steps": num_steps}

    def update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """
        Meta-update using a batch of tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y)
        """
        # Move to device
        processed_tasks = []
        for task in tasks:
            sx, sy, qx, qy = task
            processed_tasks.append((
                sx.to(self.device), sy.to(self.device),
                qx.to(self.device), qy.to(self.device)
            ))

        # Handle stateful issue similarly to adapt
        initial_hidden = None
        # If stateful, we need initial_hidden.

        try:
            loss = self.meta_maml.meta_update(
                processed_tasks, initial_hidden_state=initial_hidden
            )
        except ValueError as e:
             return {"status": "failed", "error": str(e)}

        return {"loss": loss}

    def reset_hidden(self, hidden_state: torch.Tensor):
        """Manually set hidden state (required for stateful models)."""
        self.hidden_state = hidden_state
