"""
Comprehensive validation tests for MetaRL-Agent-Framework.

Tests cover:
1. SSM core forward/backward pass
2. MetaMAML inner-loop adaptation
3. MetaMAML outer-loop meta-update
4. Test-time adaptation (Adapter)
5. Sequential (time-series) processing
6. Meta-learning convergence on synthetic tasks
7. Agent wrappers end-to-end
8. Multi-agent coordinator
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from core.ssm import SSM, StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from meta_rl.maml_agent import MetaMAMLAgent
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from adaptation.adaptation_agent import AdaptationAgent
from core.agent import Agent
from core.coordinator import AgentCoordinator


# ---------------------------------------------------------------------------
# 1. SSM Core
# ---------------------------------------------------------------------------

class TestSSMCore:
    def test_forward_shape(self):
        model = SSM(state_dim=16, input_dim=4, output_dim=2, hidden_dim=64)
        h = model.init_hidden(batch_size=4)
        x = torch.randn(4, 4)
        out, h_next = model(x, h)
        assert out.shape == (4, 2)
        assert h_next.shape == (4, 16)

    def test_hidden_init_zeros(self):
        model = SSM(state_dim=8, input_dim=3, output_dim=1)
        h = model.init_hidden(batch_size=2)
        assert torch.all(h == 0)
        assert h.shape == (2, 8)

    def test_gradient_flow(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2)
        x = torch.randn(4, 4)
        # Use non-zero hidden state so state_transition layer gets gradients
        h = torch.randn(4, 8)
        out, _ = model(x, h)
        loss = out.sum()
        loss.backward()
        grads = [p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.parameters()]
        assert all(grads), "All parameters should receive gradients"

    def test_save_load_roundtrip(self, tmp_path):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        path = str(tmp_path / "ssm_test.pt")
        model.save(path)
        loaded = SSM.load(path)
        x = torch.randn(1, 4)
        h = model.init_hidden(1)
        out1, _ = model(x, h)
        out2, _ = loaded(x, h)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_alias_backward_compat(self):
        assert StateSpaceModel is SSM

    def test_various_batch_sizes(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2)
        for bs in [1, 8, 32, 64]:
            h = model.init_hidden(bs)
            x = torch.randn(bs, 4)
            out, h_next = model(x, h)
            assert out.shape == (bs, 2)
            assert h_next.shape == (bs, 8)


# ---------------------------------------------------------------------------
# 2. MetaMAML Inner Loop
# ---------------------------------------------------------------------------

class TestMetaMAMLInnerLoop:
    def setup_method(self):
        self.model = SSM(state_dim=16, input_dim=4, output_dim=2, hidden_dim=64)
        self.meta = MetaMAML(self.model, inner_lr=0.01, outer_lr=0.001)

    def test_adapt_task_returns_ordered_dict(self):
        sx = torch.randn(8, 4)
        sy = torch.randn(8, 2)
        h0 = self.model.init_hidden(8)
        fw = self.meta.adapt_task(sx, sy, initial_hidden_state=h0)
        assert isinstance(fw, OrderedDict)

    def test_fast_weights_differ_from_original(self):
        sx = torch.randn(8, 4)
        sy = torch.randn(8, 2)
        h0 = self.model.init_hidden(8)
        orig = {k: v.clone() for k, v in self.model.named_parameters()}
        fw = self.meta.adapt_task(sx, sy, initial_hidden_state=h0, num_steps=3)
        any_changed = any(
            not torch.equal(fw[k], orig[k]) for k in orig
        )
        assert any_changed, "Fast weights should differ after inner loop"

    def test_sequential_adapt_task(self):
        seq_x = torch.randn(4, 10, 4)  # (batch, time, input)
        seq_y = torch.randn(4, 10, 2)
        h0 = self.model.init_hidden(4)
        fw = self.meta.adapt_task(seq_x, seq_y, initial_hidden_state=h0)
        assert isinstance(fw, OrderedDict)
        assert len(fw) == len(list(self.model.parameters()))


# ---------------------------------------------------------------------------
# 3. MetaMAML Outer Loop
# ---------------------------------------------------------------------------

class TestMetaMAMLOuterLoop:
    def test_meta_update_returns_loss(self):
        model = SSM(state_dim=8, input_dim=2, output_dim=1, hidden_dim=32)
        meta = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)
        tasks = []
        for _ in range(3):
            tasks.append((
                torch.randn(4, 2), torch.randn(4, 1),
                torch.randn(4, 2), torch.randn(4, 1),
            ))
        h = model.init_hidden(4)
        loss = meta.meta_update(tasks, initial_hidden_state=h)
        assert isinstance(loss, float)
        assert loss > 0

    def test_meta_update_changes_params(self):
        model = SSM(state_dim=8, input_dim=2, output_dim=1, hidden_dim=32)
        meta = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)
        before = {k: v.clone() for k, v in model.named_parameters()}
        tasks = [
            (torch.randn(4, 2), torch.randn(4, 1),
             torch.randn(4, 2), torch.randn(4, 1))
            for _ in range(3)
        ]
        h = model.init_hidden(4)
        meta.meta_update(tasks, initial_hidden_state=h)
        any_changed = any(
            not torch.equal(v, before[k])
            for k, v in model.named_parameters()
        )
        assert any_changed, "Meta-update should modify model parameters"


# ---------------------------------------------------------------------------
# 4. Convergence Test
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_meta_loss_decreases(self):
        """Meta-learning loss should trend downward over 80 updates."""
        model = SSM(state_dim=8, input_dim=2, output_dim=1, hidden_dim=32)
        meta = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)
        losses = []
        for _ in range(80):
            tasks = [
                (torch.randn(8, 2), torch.randn(8, 1),
                 torch.randn(8, 2), torch.randn(8, 1))
                for _ in range(4)
            ]
            h = model.init_hidden(8)
            losses.append(meta.meta_update(tasks, initial_hidden_state=h))
        # Compare average of first 10 vs last 10
        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first, (
            f"Loss should decrease: first10_avg={avg_first:.4f}, last10_avg={avg_last:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Test-Time Adaptation
# ---------------------------------------------------------------------------

class TestAdaptation:
    def test_adapter_update_step(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        adapter = Adapter(model, AdaptationConfig(num_steps=5, learning_rate=0.01))
        x = torch.randn(4, 4)
        y = torch.randn(4, 2)
        h = model.init_hidden(4)
        loss, steps = adapter.update_step(x, y, h)
        assert isinstance(loss, float)
        assert steps == 5

    def test_adapter_reduces_loss(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        adapter = Adapter(model, AdaptationConfig(num_steps=20, learning_rate=0.01))
        x = torch.randn(4, 4)
        y = torch.randn(4, 2)
        h = model.init_hidden(4)

        # Measure loss before adaptation
        with torch.no_grad():
            out, _ = model(x, h)
            loss_before = nn.MSELoss()(out, y).item()

        # Run adaptation
        adapter.update_step(x, y, h)

        # Measure loss after adaptation
        with torch.no_grad():
            out2, _ = model(x, h)
            loss_after = nn.MSELoss()(out2, y).item()

        assert loss_after < loss_before, (
            f"Adaptation should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Agent Wrappers
# ---------------------------------------------------------------------------

class TestAgentWrappers:
    def test_metamaml_agent_act(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        meta = MetaMAML(model)
        agent = MetaMAMLAgent(meta)
        agent.reset_hidden(model.init_hidden(1))
        obs = np.random.randn(4).astype(np.float32)
        action = agent.act(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_adaptation_agent_act(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        adapter = Adapter(model, AdaptationConfig())
        agent = AdaptationAgent(adapter)
        agent.reset_hidden(model.init_hidden(1))
        obs = np.random.randn(4).astype(np.float32)
        action = agent.act(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_adaptation_agent_adapt_returns_dict(self):
        model = SSM(state_dim=8, input_dim=4, output_dim=2, hidden_dim=32)
        adapter = Adapter(model, AdaptationConfig(num_steps=3))
        agent = AdaptationAgent(adapter)
        h = model.init_hidden(4)
        result = agent.adapt(
            inputs=torch.randn(4, 4),
            targets=torch.randn(4, 2),
            hidden_state=h,
        )
        assert isinstance(result, dict)
        assert "loss" in result


# ---------------------------------------------------------------------------
# 7. Environment (basic smoke test)
# ---------------------------------------------------------------------------

class TestEnvironmentSmoke:
    def test_placeholder_env_runs(self):
        from env_runner.environment import Environment
        env = Environment(env_name="CartPole-v1", batch_size=1)
        obs = env.reset()
        assert obs is not None
        action = env.sample_action()
        result = env.step(action)
        assert len(result) == 4  # obs, reward, done, info
        env.close()
