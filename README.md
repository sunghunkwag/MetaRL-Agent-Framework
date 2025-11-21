# MetaRL-Agent-Framework


[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-automated-blue)](https://github.com/users/sunghunkwag/packages/container/package/ssm-metarl-testcompute)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/demo.ipynb)

## Features

- **Unified Agent Abstraction**: Clean, unified `Agent` interface with `act`, `observe`, `update`, and `adapt`.
- **Agent Wrappers**: `MetaMAMLAgent` and `AdaptationAgent` wrappers for easy integration.
- **Agent wrappers**: `MetaMAMLAgent` and `AdaptationAgent` wrappers for easy integration.
- **Multi-Agent Coordination**: Basic coordinator to run multiple agents in parallel or sequence.
- **State Space Models (SSM)** for temporal dynamics modeling
- **Meta-Learning (MAML)** for fast adaptation across tasks
- **Test-Time Adaptation** for online model improvement
- **Modular Architecture** with clean, testable components
- **Gymnasium Integration** for RL environment compatibility
- **Test Suite** with automated CI/CD
- **Docker Container** ready for deployment
- **High-dimensional Benchmarks** with MuJoCo tasks and baseline comparisons

## Project Structure

- **core/**: Core model implementations and Agent abstractions
  - `agent.py`: Base `Agent` class defining the unified interface.
  - `coordinator.py`: `AgentCoordinator` for managing multiple agents.
  - `ssm.py`: State Space Model implementation.
- **meta_rl/**: Meta-learning algorithms
  - `maml_agent.py`: `MetaMAMLAgent` wrapper for MAML.
  - `meta_maml.py`: MetaMAML implementation.
- **adaptation/**: Test-time adaptation
  - `adaptation_agent.py`: `AdaptationAgent` wrapper for Adapter.
  - `test_time_adaptation.py`: Adapter class.
- **env_runner/**: Environment utilities
  - `environment.py`: Gymnasium environment wrapper
- **examples/**: Example scripts
  - `multi_agent_demo.py`: Demonstration of multi-agent coordination.
- **experiments/**: Experiment scripts and benchmarks
  - `quick_benchmark.py`: Quick benchmark suite.
  - `serious_benchmark.py`: High-dimensional MuJoCo benchmarks.

## Agent System

The framework now supports a unified Agent API (`act`, `observe`, `adapt`, `update`) to allow seamless interchange of different learning strategies.

### Base Agent

The `Agent` class in `core/agent.py` defines the contract:
- `act(observation) -> action`
- `observe(observation, action, reward, ...) -> None`
- `adapt(...) -> info`: Inner-loop adaptation
- `update(...) -> info`: Outer-loop meta-update

### Wrappers

- **MetaMAMLAgent**: Wraps `MetaMAML` for meta-learning tasks.
- **AdaptationAgent**: Wraps `Adapter` for test-time adaptation.

### Usage Example


The framework now supports a unified Agent API (`act`, `observe`, `adapt`, `update`) to allow seamless interchange of different learning strategies.

### Base Agent

The `Agent` class in `core/agent.py` defines the contract:
- `act(observation) -> action`
- `observe(observation, action, reward, ...) -> None`
- `adapt(...) -> info`: Inner-loop adaptation
- `update(...) -> info`: Outer-loop meta-update

### wrappers

- **MetaMAMLAgent**: Wraps `MetaMAML` for meta-learning tasks.
- **AdaptationAgent**: Wraps `Adapter` for test-time adaptation.

### Usage Example

```python
from core.coordinator import AgentCoordinator
from meta_rl.maml_agent import MetaMAMLAgent
from adaptation.adaptation_agent import AdaptationAgent

# ... Setup models ...

agent_maml = MetaMAMLAgent(maml_algo)
agent_adapt = AdaptationAgent(adapter)

coordinator = AgentCoordinator()
coordinator.register_agent(agent_maml, "MAML")
coordinator.register_agent(agent_adapt, "Adapt")

# Run agents
observations = env.reset()
actions = coordinator.run_step_all(observations)
```

See `examples/multi_agent_demo.py` for a complete runnable example.

## Interactive Demo

**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/demo.ipynb)


**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/demo.ipynb)

Run the complete demo in your browser with Google Colab - no installation required!

---

## Quick Start (Simple Demo)

### Installation

```bash
git clone https://github.com/sunghunkwag/SSM-MetaRL-TestCompute.git
cd SSM-MetaRL-TestCompute
pip install -e .
```

### Run Multi-Agent Demo

```bash
python examples/multi_agent_demo.py
```

### Run Main Script

```bash
# Train on CartPole environment
python main.py --env_name CartPole-v1 --num_epochs 20
```

## Core Components

### State Space Model (SSM)
... (Same as before)

### MetaMAML
... (Same as before)

### Adapter (Test-Time Adaptation)
... (Same as before)

### Environment Runner

The `Environment` class in `env_runner/environment.py` provides a wrapper around Gymnasium environments.

**Key Features**:
- Simplified API: `reset()` returns only observation (not tuple)
- Simplified API: `step(action)` returns 4 values (obs, reward, done, info)
- Batch processing support with `batch_size` parameter
... (Same as before)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Gymnasium >= 1.0
- NumPy
- pytest (for development)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ssm_metarl_testcompute,
  author = {sunghunkwag},
  title = {SSM-MetaRL-TestCompute: A Framework for Meta-RL with State Space Models},
  year = {2025},
  url = {https://github.com/sunghunkwag/SSM-MetaRL-TestCompute}
}
```

## Acknowledgments

This framework builds upon research in:
- State Space Models for sequence modeling
- Model-Agnostic Meta-Learning (MAML)
- Test-time adaptation techniques
- Reinforcement learning with Gymnasium
