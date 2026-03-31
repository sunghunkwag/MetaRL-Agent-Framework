# MetaRL-Agent-Framework

[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen)](https://github.com/sunghunkwag/MetaRL-Agent-Framework)
[![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Docker Build](https://github.com/sunghunkwag/MetaRL-Agent-Framework/actions/workflows/publish-docker.yml/badge.svg)](https://github.com/sunghunkwag/MetaRL-Agent-Framework/actions/workflows/publish-docker.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-available-blue)](https://github.com/sunghunkwag/MetaRL-Agent-Framework/pkgs/container/metarl-agent-framework)

A modular meta-reinforcement learning framework combining **State Space Models (SSM)**, **MAML**, and **test-time adaptation** with a unified agent interface.

## Quick Start

```bash
git clone https://github.com/sunghunkwag/MetaRL-Agent-Framework.git
cd MetaRL-Agent-Framework
pip install -e ".[dev]"

# Run tests (40 tests, all passing)
pytest tests/ -v

# Train on CartPole
python main.py --env_name CartPole-v1 --num_epochs 20
```

## Architecture

```
core/          SSM model, abstract Agent, AgentCoordinator
meta_rl/       MetaMAML (inner/outer loop), MetaMAMLAgent wrapper
adaptation/    Test-time Adapter, AdaptationAgent wrapper
env_runner/    Gymnasium environment wrapper (batch support)
tests/         40 unit + integration + convergence tests
```

### Key Components

| Component | Description |
|---|---|
| **SSM** (`core/ssm.py`) | State space model: $x_{t+1} = Ax_t + Bu_t$, $y_t = Cx_t + Du_t$ |
| **MetaMAML** (`meta_rl/meta_maml.py`) | MAML with `torch.func.functional_call`, supports stateful models |
| **Adapter** (`adaptation/test_time_adaptation.py`) | Online test-time adaptation with configurable steps/lr |
| **Agent** (`core/agent.py`) | Unified interface: `act`, `observe`, `adapt`, `update` |
| **Coordinator** (`core/coordinator.py`) | Multi-agent parallel/sequential execution |

## Usage

```python
from core.ssm import SSM
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig

# Create model + meta-learner
model = SSM(state_dim=16, input_dim=4, output_dim=2)
meta = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

# Inner-loop adaptation
fast_weights = meta.adapt_task(support_x, support_y, initial_hidden_state=h0)

# Outer-loop meta-update
loss = meta.meta_update(tasks, initial_hidden_state=h0)

# Test-time adaptation
adapter = Adapter(model, AdaptationConfig(num_steps=5, learning_rate=0.01))
loss, steps = adapter.update_step(x, y, hidden_state)
```

## Validation Results (2026-03-31)

All **40 tests passed** across 5 test files:

| Suite | Tests | Status |
|---|---|---|
| SSM Core (forward, gradient, save/load, batching) | 10 | Pass |
| MetaMAML Inner/Outer Loop | 3 | Pass |
| Test-Time Adaptation | 6 | Pass |
| Agent Wrappers | 3 | Pass |
| Comprehensive Validation (convergence, loss reduction) | 18 | Pass |

Key findings:
- Meta-learning loss converges (avg first 10 epochs > avg last 10 epochs)
- Adapter measurably reduces loss on held-out data
- Fast weights diverge from meta-parameters after inner loop (adaptation works)
- Sequential (time-series) processing verified for both MAML and SSM

## Benchmarks

Supports MuJoCo, Meta-World ML10/ML45, and iMuJoCo via `--benchmark-mode`:

```bash
# Few-shot adaptation
python main.py --task-dist metaworld-ml10 --method ssm --benchmark-mode --few-shot 1 5 10

# Zero-shot generalization
python main.py --task-dist imujoco-halfcheetah --method ssm --benchmark-mode --zero-shot
```

## Docker

```bash
docker build -t metarl-agent-framework .
docker run -it metarl-agent-framework
```

## Dependencies

- Python >= 3.10, PyTorch >= 2.0, Gymnasium >= 0.26
- Dev: pytest, black, flake8

## License

MIT - see [LICENSE](./LICENSE).
