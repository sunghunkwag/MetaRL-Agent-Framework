import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from core.agent import Agent
from adaptation.test_time_adaptation import Adapter, AdaptationConfig

class AdaptationAgent(Agent):
    """
    Agent wrapper for Test-Time Adaptation (Adapter).

    Handles:
    - Observation storage
    - Interaction (act) using the underlying model
    - Test-time adaptation (adapt)
    """

    def __init__(self,
                 adapter: Adapter,
                 device: str = 'cpu'):
        self.adapter = adapter
        self.device = device
        self.hidden_state = None

        # Check if model is stateful (simple check similar to MetaMAML)
        import inspect
        sig = inspect.signature(self.adapter.model.forward)
        self._stateful = 'hidden_state' in sig.parameters or len(sig.parameters) > 1 # Heuristic

    def reset(self):
        self.hidden_state = None

    def reset_hidden(self, hidden_state: torch.Tensor):
        self.hidden_state = hidden_state

    def act(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Select action using current model weights."""
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float().to(self.device)
        else:
            obs_tensor = observation.to(self.device)

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            self.adapter.model.eval()
            if self._stateful:
                if self.hidden_state is None:
                     raise RuntimeError("Stateful model requires hidden_state. Call reset_hidden().")
                output, next_hidden = self.adapter.model(obs_tensor, self.hidden_state)
                self.hidden_state = next_hidden
            else:
                # Assume forward(x)
                output = self.adapter.model(obs_tensor)

        return output.cpu().numpy().squeeze(0)

    def observe(self, observation: Any, action: Any, reward: float, terminated: bool, truncated: bool, info: Dict) -> None:
        # AdaptationAgent might process data immediately in adapt()
        # or store it. The Adapter.update_step takes x, y, hidden.
        # We'll assume the user calls adapt() explicitly with data.
        pass

    def adapt(self, inputs: torch.Tensor, targets: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform test-time adaptation step.

        Args:
            inputs: Input tensor (x)
            targets: Target tensor (y)
            hidden_state: Hidden state for the update step (if stateful)
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if hidden_state is None and self._stateful:
            hidden_state = self.hidden_state

        loss, steps = self.adapter.update_step(inputs, targets, hidden_state)

        return {"loss": loss, "steps": steps}

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Outer loop update.
        For pure test-time adaptation, this is typically not used or just logs status.
        """
        return {"status": "no_op", "message": "AdaptationAgent does not support meta-training update."}
