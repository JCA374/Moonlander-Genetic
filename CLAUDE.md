# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **Genetic Algorithm (GA)** approach to solving the Lunar Lander environment using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to evolve a simple linear controller with only 36 parameters.

## Essential Commands

### Training

```bash
# Standard training (~20-40 minutes, 200 generations)
python train_ga.py

# Quick test (faster, for development)
python train_ga.py --generations 50 --population 30 --episodes 5

# High performance with parallelization (faster on multi-core)
python train_ga.py --parallel 4

# Thorough training
python train_ga.py --generations 500 --population 100 --episodes 20
```

### Evaluation and Visualization

```bash
# Play trained GA model
python play_ga.py --model ga_models/ga_best.npy --episodes 5

# Describe what controller learned (interpretability!)
python play_ga.py --model ga_models/ga_best.npy --describe

# Evaluate performance over many episodes
python play_ga.py --model ga_models/ga_best.npy --evaluate 100 --no-render

# Plot training progress
python plot_ga_training.py --history ga_models/training_history.json
```

### Testing

```bash
# Test GA components (works without Box2D)
python test_ga_components.py

# Integration test for full GA pipeline
python test_integration.py
```

## Architecture

### Core Components

**GA System** (`train_ga.py`):
- `linear_controller.py`: Simple 36-parameter linear policy (8 states → 4 actions)
- `ga_evaluator.py`: Fitness evaluation (avg_reward + landing_rate × 100)
- `cma_es_optimizer.py`: CMA-ES optimizer with simple GA fallback if `cma` package unavailable
- `train_ga.py`: Main training loop with checkpointing and history tracking
- `play_ga.py`: Visualization and evaluation
- `plot_ga_training.py`: Training progress plotting

### Linear Controller Design

The GA approach uses a simple, interpretable linear mapping:
```
action_scores = state @ weights + bias
action = argmax(action_scores)
```

Where:
- **State**: 8-dimensional (x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact)
- **Weights**: 8×4 matrix (32 parameters)
- **Bias**: 4-dimensional vector (4 parameters)
- **Actions**: [0=nothing, 1=left engine, 2=main engine, 3=right engine]
- **Total parameters**: 36

This simplicity enables:
- Direct inspection of learned weights
- Understanding which state features drive which actions
- Fast evaluation and parallel training
- Interpretability - you can see exactly what the controller learned

### CMA-ES Optimization

CMA-ES is a state-of-the-art evolutionary algorithm that:

1. **Generates** a population of candidate solutions (parameter vectors)
2. **Evaluates** each by running episodes in the environment
3. **Updates** the search distribution based on fitness
4. **Adapts** the covariance matrix to learn problem structure
5. **Repeats** until convergence or max generations

**Fitness Function:**
```python
fitness = avg_reward + (landing_rate × 100)
```

This encourages both high episode rewards and successful landings.

### Model Saving

Training saves:
- `ga_best.npy`: Best controller found (updated whenever fitness improves)
- `ga_checkpoint_{generation:04d}.npy`: Checkpoints every 20 generations (configurable)
- `training_history.json`: Complete training history for plotting

## Dependencies

```bash
# Core requirements
pip install -r requirements.txt

# Optional: For optimal CMA-ES performance
pip install cma
# If not installed, simple GA fallback is used automatically
```

## Directory Structure

- `ga_models/`: Saved GA models and training history (gitignored)
- `videos/`: Generated episode videos (gitignored)
- `old/`: Archived code (gitignored)
- `tests/`: Test files (test_ga_components.py works without Box2D)
- `publish/`: Published visualizations

## Development Notes

### When working with GA:

1. **CMA-ES Package**: Requires `cma` package for best performance, but has fallback to simple GA if not installed

2. **Parallel Evaluation**: Use `--parallel N` for near-linear speedup on multi-core CPUs
   ```bash
   python train_ga.py --parallel 4  # 4x speedup on 4+ core CPU
   ```

3. **Fitness Function**: Combines reward and landing success in `ga_evaluator.py:evaluate_controller()`
   ```python
   fitness = avg_reward + (landing_rate × 100)
   ```
   Can be customized to prioritize different objectives.

4. **Interpretability**: Use `--describe` flag to understand learned weights
   ```bash
   python play_ga.py --model ga_models/ga_best.npy --describe
   ```
   This shows which state features most influence each action.

5. **Hyperparameters** (in `train_ga.py`):
   - `--generations`: Number of evolution cycles (default: 200)
   - `--population`: Population size (default: 50)
   - `--episodes`: Episodes per fitness evaluation (default: 10)
   - `--sigma`: Initial step size (default: 0.5)
   - `--parallel`: Number of parallel workers (default: 1)

### Expected Training Results

With standard settings (200 generations, population 50):
- **Generation 0:** ~-200 fitness, 0-10% landing rate (random)
- **Generation 50:** ~50-100 fitness, 20-40% landing rate
- **Generation 100:** ~150-200 fitness, 50-70% landing rate
- **Generation 200:** ~200-250 fitness, 70-90% landing rate

Best controllers can achieve:
- **Fitness:** 250-300
- **Landing Rate:** 80-95%
- **Average Reward:** 180-220

### Debugging Training

If no improvement:
- Try increasing sigma: `--sigma 1.0`
- Increase population: `--population 100`
- More episodes per eval: `--episodes 20`

If training too slow:
- Reduce episodes: `--episodes 5`
- Smaller population: `--population 30`
- Use parallelization: `--parallel 4`

### Advantages of GA Approach

1. **Simplicity**: Much simpler than deep RL (no replay buffer, target networks, etc.)
2. **Interpretability**: Can inspect and understand learned weights
3. **Robustness**: Less sensitive to hyperparameters
4. **Parallelization**: Easy to distribute across CPU cores
5. **No gradients**: Works with any controller structure
6. **Small model**: Only 36 parameters vs thousands in neural networks
