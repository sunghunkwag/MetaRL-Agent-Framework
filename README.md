# SSM-MetaRL-TestCompute

A research framework combining State Space Models (SSM), Meta-Learning (MAML), and Test-Time Adaptation for reinforcement learning.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sunghunkwag/SSM-MetaRL-TestCompute)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-automated-blue)](https://github.com/users/sunghunkwag/packages/container/package/ssm-metarl-testcompute)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/demo.ipynb)

## Features

- **Unified Agent Abstraction**: Clean, unified `Agent` interface with `act`, `observe`, `update`, and `adapt`.
- **Agent Wrappers**: `MetaMAMLAgent` and `AdaptationAgent` wrappers for easy integration.
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

The SSM implementation in `core/ssm.py` models state transitions.

**API**:
- `forward(x, hidden_state)` returns a tuple: `(output, next_hidden_state)`.
- `init_hidden(batch_size)` provides the initial hidden state.

Constructor Arguments:
- `state_dim` (int): Internal state dimension
- `input_dim` (int): Input feature dimension
- `output_dim` (int): Output feature dimension
- `hidden_dim` (int): Hidden layer dimension within networks
- `device` (str or torch.device)

Example usage:
```python
import torch
from core.ssm import StateSpaceModel

model = StateSpaceModel(state_dim=128, input_dim=64, output_dim=32, device='cpu')
batch_size = 4
input_x = torch.randn(batch_size, 64)
current_hidden = model.init_hidden(batch_size)

# Forward pass requires current state and returns next state
output, next_hidden = model(input_x, current_hidden)
print(output.shape)       # torch.Size([4, 32])
print(next_hidden.shape)  # torch.Size([4, 128])
```

### MetaMAML

The `MetaMAML` class in `meta_rl/meta_maml.py` implements MAML.

**Key Features**:
- Handles **stateful models** (like SSM)
- Supports **time series input** `(B, T, D)`
- **API**: `meta_update` takes `tasks` (a list of tuples) and `initial_hidden_state` as arguments

**Time Series Input Handling**:
Input data should be shaped `(batch_size, time_steps, features)`. MAML processes sequences internally.

Example with time series:

```python
import torch
import torch.nn.functional as F
from meta_rl.meta_maml import MetaMAML
from core.ssm import StateSpaceModel

model = StateSpaceModel(state_dim=64, input_dim=32, output_dim=16, device='cpu')
maml = MetaMAML(model, inner_lr=0.01, outer_lr=0.001)

# Time series input: (batch=4, time_steps=10, features=32)
support_x = torch.randn(4, 10, 32)
support_y = torch.randn(4, 10, 16)
query_x = torch.randn(4, 10, 32)
query_y = torch.randn(4, 10, 16)

# Prepare tasks as a list of tuples
tasks = []
for i in range(4):
    tasks.append((support_x[i:i+1], support_y[i:i+1], query_x[i:i+1], query_y[i:i+1]))

# Initialize hidden state
initial_hidden = model.init_hidden(batch_size=4)

# Call meta_update with tasks list and initial state
loss = maml.meta_update(tasks=tasks, initial_hidden_state=initial_hidden, loss_fn=F.mse_loss)
print(f"Meta Loss: {loss:.4f}")
```

Constructor Arguments:
- `model`: The base model.
- `inner_lr` (float): Inner loop learning rate.
- `outer_lr` (float): Outer loop learning rate.
- `first_order` (bool): Use first-order MAML.

### Adapter (Test-Time Adaptation)

The `Adapter` class in `adaptation/test_time_adaptation.py` performs test-time adaptation.

**Key Features**:
- **API**: `update_step` takes `x`, `y` (target), and `hidden_state` directly as arguments
- Internally performs `config.num_steps` gradient updates per call
- Properly detaches hidden state to prevent autograd computational graph errors
- Manages hidden state across internal steps
- Returns `(loss, steps_taken)`

Constructor Arguments:
- `model`: The model to adapt.
- `config`: An `AdaptationConfig` object containing `learning_rate` and `num_steps`.
- `device`: Device string ('cpu' or 'cuda').

Example usage:

```python
import torch
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import StateSpaceModel

# Model output dim must match target 'y'
model = StateSpaceModel(state_dim=64, input_dim=32, output_dim=32, device='cpu')
config = AdaptationConfig(learning_rate=0.01, num_steps=5)
adapter = Adapter(model=model, config=config, device='cpu')

# Initialize hidden state
hidden_state = model.init_hidden(batch_size=1)

# Adaptation loop
for step in range(10):
    x = torch.randn(1, 32)
    y_target = torch.randn(1, 32)
    
    # Store current state for adaptation call
    current_hidden_state_for_adapt = hidden_state
    
    # Get next state prediction (optional)
    with torch.no_grad():
        output, hidden_state = model(x, current_hidden_state_for_adapt)
    
    # Call update_step with x, target, and state_t
    loss, steps_taken = adapter.update_step(
        x=x,
        y=y_target,
        hidden_state=current_hidden_state_for_adapt
    )
    print(f"Adapt Call {step}, Loss: {loss:.4f}, Internal Steps: {steps_taken}")
```

### Environment Runner

The `Environment` class in `env_runner/environment.py` provides a wrapper around Gymnasium environments.

**Key Features**:
- Simplified API: `reset()` returns only observation (not tuple)
- Simplified API: `step(action)` returns 4 values (obs, reward, done, info)
- Batch processing support with `batch_size` parameter

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
