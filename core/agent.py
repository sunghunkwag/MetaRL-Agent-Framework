from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

class Agent(ABC):
    """
    Abstract base class for all agents in the system.

    This defines the unified interface for agents, regardless of their
    underlying implementation (e.g., Meta-Learning, Test-Time Adaptation).
    """

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Select an action based on the given observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            The selected action.
        """
        pass

    @abstractmethod
    def observe(self, observation: Any, action: Any, reward: float, terminated: bool, truncated: bool, info: Dict) -> None:
        """
        Observe the result of an action.

        This method is typically used to update internal buffers or state
        tracking, but does not trigger a model update/adaptation itself.

        Args:
            observation: The observation *after* the action was taken.
            action: The action that was taken.
            reward: The reward received.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated (time limit).
            info: Additional information from the environment.
        """
        pass

    @abstractmethod
    def adapt(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform an inner-loop adaptation or test-time update.

        For meta-learning agents, this updates the "fast weights" based on
        recent experience (support set).

        Returns:
            A dictionary containing metrics or status of the adaptation.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform an outer-loop update (meta-training).

        For meta-learning agents, this updates the meta-parameters.
        For test-time agents, this might be a no-op.

        Returns:
            A dictionary containing metrics (e.g., loss) of the update.
        """
        pass
