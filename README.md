# MetaRL-Agent-Framework

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sunghunkwag/MetaRL-Agent-Framework)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Docker](https://img.shields.io/badge/docker-automated-blue)](https://github.com/users/sunghunkwag/packages/container/package/metarl-agent-framework)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/MetaRL-Agent-Framework/blob/main/demo.ipynb)

A modular meta-reinforcement learning framework with unified agent abstractions, multi-agent coordination, and extensible adaptation mechanisms.

## Features

- **Unified Agent Abstraction**: Clean, unified `Agent` interface with `act`, `observe`, `update`, and `adapt` methods
- **Agent Wrappers**: `MetaMAMLAgent` and `AdaptationAgent` wrappers for seamless integration
- **Multi-Agent Coordination**: Basic coordinator to run multiple agents in parallel or sequence
- **State Space Models (SSM)**: Temporal dynamics modeling with efficient state representation
- **Meta-Learning (MAML)**: Fast adaptation across tasks using Model-Agnostic Meta-Learning
- **Test-Time Adaptation**: Online model improvement during deployment
- **Modular Architecture**: Clean, testable, and extensible components
- **Gymnasium Integration**: Full compatibility with OpenAI Gymnasium environments
- **Automated Testing**: Comprehensive test suite with CI/CD integration
- **Docker Support**: Production-ready containerized deployment
- **High-Dimensional Benchmarks**: MuJoCo, Meta-World, and iMuJoCo support

## Advanced Benchmarks

The framework includes support for 4 major high-dimensional benchmark suites (10 distributions total):

1.  **Standard MuJoCo**:
    *   `halfcheetah-vel`, `ant-vel`, `walker2d-vel` (Velocity control)
    *   `ant-dir` (Direction control)
    *   `halfcheetah-gravity`, `ant-mass` (Dynamics randomization)

2.  **Meta-World (v2)**:
    *   `metaworld-ml10`: 10 train / 5 test manipulation tasks (39-dim obs)
    *   `metaworld-ml45`: 45 train / 5 test manipulation tasks

3.  **iMuJoCo**:
    *   `imujoco-halfcheetah`: Robustness/generalization benchmarks
    *   `imujoco-ant`: Complex morphology generalization

### Running Benchmarks

The `main.py` script supports a dedicated `--benchmark-mode` for rigorous evaluation.

#### 1. Few-Shot Adaptation
Train on meta-train tasks, then evaluate adaptation performance on held-out test tasks given `K` shots (gradient steps).

```bash
# Meta-World ML10 with 1, 5, and 10 shots
python main.py --task-dist metaworld-ml10 \
               --method ssm \
               --benchmark-mode \
               --few-shot 1 5 10 \
               --epochs 50
```

#### 2. Zero-Shot Generalization
Train on meta-train tasks, then evaluate directly on held-out test tasks without adaptation.

```bash
# iMuJoCo HalfCheetah Zero-Shot with custom split
python main.py --task-dist imujoco-halfcheetah \
               --method ssm \
               --benchmark-mode \
               --zero-shot \
               --train-tasks 0 1 2 \
               --test-tasks 3 4 \
               --epochs 20
```

#### 3. Standard Benchmark Run
Run meta-training with periodic evaluation on held-out tasks.

```bash
# Standard run
python main.py --task-dist metaworld-ml45 \
               --method ssm \
               --benchmark-mode \
               --epochs 100
```

## Project Structure

```
MetaRL-Agent-Framework/
├── core/                  # Core implementations
│   ├── agent.py          # Base Agent class
│   ├── coordinator.py    # AgentCoordinator for multi-agent
│   └── ssm.py            # State Space Model
├── meta_rl/              # Meta-learning algorithms
│   ├── maml_agent.py     # MetaMAMLAgent wrapper
│   └── meta_maml.py      # MetaMAML implementation
├── adaptation/           # Test-time adaptation
│   ├── adaptation_agent.py    # AdaptationAgent wrapper
│   └── test_time_adaptation.py # Adapter class
├── env_runner/           # Environment utilities
│   └── environment.py    # Gymnasium wrapper
├── examples/             # Example scripts
│   └── multi_agent_demo.py    # Multi-agent demo
├── experiments/          # Experiment scripts
│   ├── quick_benchmark.py     # Quick benchmarks
│   ├── serious_benchmark.py   # Full benchmark suite
│   ├── task_distributions.py  # Benchmark definitions
│   └── test_integration.py    # End-to-end tests
└── tests/                # Test suite
```

## Agent System

The framework provides a unified Agent API for seamless interchange of different learning strategies.

### Base Agent

The `Agent` class in `core/agent.py` defines the contract:

- `act(observation) -> action`: Select action based on observation
- `observe(observation, action, reward, ...) -> None`: Record experience
- `adapt(...) -> info`: Inner-loop adaptation (task-specific learning)
- `update(...) -> info`: Outer-loop meta-update (cross-task learning)

### Agent Wrappers

- **MetaMAMLAgent**: Wraps `MetaMAML` for meta-learning tasks
- **AdaptationAgent**: Wraps `Adapter` for test-time adaptation

### Multi-Agent Coordination

The `AgentCoordinator` in `core/coordinator.py` enables:
- Parallel execution of multiple agents
- Sequential task distribution
- Centralized observation and action management

### Usage Example

```python
from core.coordinator import AgentCoordinator
from meta_rl.maml_agent import MetaMAMLAgent
from adaptation.adaptation_agent import AdaptationAgent

# Create agents
agent_maml = MetaMAMLAgent(maml_algo)
agent_adapt = AdaptationAgent(adapter)

# Setup coordinator
coordinator = AgentCoordinator()
coordinator.register_agent(agent_maml, "MAML")
coordinator.register_agent(agent_adapt, "Adapt")

# Run agents
observations = env.reset()
actions = coordinator.run_step_all(observations)
```

See `examples/multi_agent_demo.py` for a complete runnable example.

## Quick Start

### Installation

```bash
git clone https://github.com/sunghunkwag/MetaRL-Agent-Framework.git
cd MetaRL-Agent-Framework
pip install -e .
```

### Run Multi-Agent Demo

```bash
python examples/multi_agent_demo.py
```

### Run Main Training Script

```bash
# Train on CartPole environment
python main.py --env_name CartPole-v1 --num_epochs 20
```

### Interactive Demo

**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/MetaRL-Agent-Framework/blob/main/demo.ipynb)

Run the complete demo in your browser with Google Colab - no installation required!

## Core Components

### State Space Model (SSM)

The `StateSpaceModel` in `core/ssm.py` provides efficient temporal dynamics modeling:

- Parameterized state transitions: $x_{t+1} = Ax_t + Bu_t$
- Observation model: $y_t = Cx_t + Du_t$
- Learnable parameters $(A, B, C, D)$
- Supports batched and sequential processing

### MetaMAML

The `MetaMAML` class in `meta_rl/meta_maml.py` implements Model-Agnostic Meta-Learning:

- **Inner loop**: Fast adaptation to new tasks via gradient descent
- **Outer loop**: Meta-optimization across task distribution
- **First-order approximation**: Efficient computation without second derivatives
- Supports arbitrary PyTorch models

### Test-Time Adaptation

The `Adapter` class in `adaptation/test_time_adaptation.py` enables online learning:

- Continual adaptation during deployment
- Minimal overhead per update step
- Compatible with any differentiable model
- Configurable adaptation rate and frequency

### Environment Runner

The `Environment` class in `env_runner/environment.py` provides a Gymnasium wrapper:

**Key Features**:
- Simplified API: `reset()` returns observation only
- Simplified API: `step(action)` returns `(obs, reward, done, info)`
- Batch processing support
- Automatic state normalization (optional)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Gymnasium >= 1.0
- NumPy
- pytest (for development)

See `pyproject.toml` for complete dependencies.

## Docker Deployment

```bash
# Build container
docker build -t metarl-agent-framework .

# Run container
docker run -it metarl-agent-framework
```

Pre-built images available at: [GitHub Container Registry](https://github.com/users/sunghunkwag/packages/container/package/metarl-agent-framework)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{metarl_agent_framework,
  author = {sunghunkwag},
  title = {MetaRL-Agent-Framework: A Modular Meta-RL Framework with Agent Abstractions},
  year = {2025},
  url = {https://github.com/sunghunkwag/MetaRL-Agent-Framework}
}
```

## Acknowledgments

This framework builds upon research in:

- State Space Models for sequence modeling
- Model-Agnostic Meta-Learning (MAML)
- Test-time adaptation techniques
- Reinforcement learning with Gymnasium

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/sunghunkwag/MetaRL-Agent-Framework/issues).
