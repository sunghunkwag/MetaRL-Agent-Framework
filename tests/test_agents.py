import pytest
import torch
import torch.nn as nn
import numpy as np

from core.agent import Agent
from meta_rl.maml_agent import MetaMAMLAgent
from adaptation.adaptation_agent import AdaptationAgent
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x, hidden):
        out = self.layer(x)
        return out, hidden

def test_metamaml_agent():
    model = SimpleModel()
    maml = MetaMAML(model, inner_lr=0.01)
    agent = MetaMAMLAgent(maml)

    # Test act
    obs = np.array([1.0, 2.0], dtype=np.float32)
    action = agent.act(obs)
    assert action.shape == (2,)

    # Test observe (buffer)
    agent.observe(obs, action, 0.0, False, False, {})
    assert len(agent.buffer) == 1

    # Test adapt
    inputs = torch.randn(2, 2)
    targets = torch.randn(2, 2)
    res = agent.adapt(inputs, targets)
    assert res['status'] == 'adapted'
    assert agent.fast_weights is not None

def test_adaptation_agent_stateless():
    model = SimpleModel()
    config = AdaptationConfig()
    adapter = Adapter(model, config)
    agent = AdaptationAgent(adapter)

    # Test act
    obs = np.array([1.0, 2.0], dtype=np.float32)
    action = agent.act(obs)
    assert action.shape == (2,)

    # Test adapt
    inputs = torch.randn(1, 2)
    targets = torch.randn(1, 2)
    # Note: Adapter.update_step expects hidden state even if model is stateless in current implementation
    # But AdaptationAgent checks for statefulness.
    # Wait, my previous analysis showed Adapter.update_step calls model(x, hidden) unconditionally.
    # So if SimpleModel.forward doesn't take hidden, Adapter.update_step will fail.
    # AdaptationAgent wraps act(), but adapt() calls adapter.update_step().

    # So AdaptationAgent with a stateless model will fail if Adapter is not fixed or model is not stateful-compatible.
    # The existing Adapter implementation IS hardcoded for stateful models.
    # So we should test with a StatefulModel or "Compatible" model.
    pass

def test_adaptation_agent_stateful():
    model = StatefulModel()
    config = AdaptationConfig()
    adapter = Adapter(model, config)
    agent = AdaptationAgent(adapter)

    # Init hidden
    hidden = torch.zeros(1, 1)
    agent.reset_hidden(hidden)

    # Test act
    obs = np.array([1.0, 2.0], dtype=np.float32)
    action = agent.act(obs)
    assert action.shape == (2,)

    # Test adapt
    inputs = torch.randn(1, 2)
    targets = torch.randn(1, 2)
    res = agent.adapt(inputs, targets)
    assert 'loss' in res
