# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements two different approaches to solving the Lunar Lander environment:
1. **DQN (Deep Q-Network)**: Neural network-based reinforcement learning (~17K parameters)
2. **Genetic Algorithm (GA)**: CMA-ES evolution of a linear controller (36 parameters)

Both approaches achieve similar performance but offer different tradeoffs in interpretability, sample efficiency, and computational requirements.

## Essential Commands

### Training

```bash
# Train with DQN (deep reinforcement learning)
python train.py

# Train with GA (evolutionary approach)
python train_ga.py --generations 200 --population 50 --episodes 10

# Quick GA test (faster, for development)
python train_ga.py --generations 50 --population 30 --episodes 5

# GA with parallelization (faster on multi-core)
python train_ga.py --parallel 4
```

### Evaluation and Visualization

```bash
# Play trained DQN model
python play_best.py

# Play trained GA model
python play_ga.py --model ga_models/ga_best.npy --episodes 5

# Describe what GA controller learned (interpretability)
python play_ga.py --model ga_models/ga_best.npy --describe

# Compare DQN vs GA performance
python compare_ga_dqn.py --episodes 100

# Plot GA training progress
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

**DQN System** (`train.py`):
- `dqn_agent.py`: Double DQN with dueling architecture (value + advantage streams)
- `reward_shaper.py`: Sophisticated reward shaping with 10+ components (see below)
- `evaluator.py`: Evaluation functions for training progress
- `logger.py`: Training logging and diagnostics

**GA System** (`train_ga.py`):
- `linear_controller.py`: Simple 36-parameter linear policy (8 states → 4 actions)
- `ga_evaluator.py`: Fitness evaluation (avg_reward + landing_rate × 100)
- `cma_es_optimizer.py`: CMA-ES optimizer with simple GA fallback if `cma` package unavailable

### Reward Shaping System

The `reward_shaper.py` module is critical to DQN performance. It implements multiple reward components with ablation control flags:

1. **Landing Zone Control** (`enable_landing_zone_control`): Prevents side engine overuse during final landing phase
   - Detects when agent is in landing zone (altitude < 0.18, distance < 0.12, speed < 0.35)
   - Applies progressive penalties to side engines as agent approaches landing
   - Provides "settling bonus" for doing nothing when well-positioned

2. **Oscillation Penalty** (`enable_oscillation_penalty`): Discourages horizontal oscillation/wobbling

3. **Commitment Bonus** (`enable_commitment_bonus`): Rewards steady descent and penalizes unnecessary corrections

4. **Speed Control** (`enable_speed_control`): Guides agent toward optimal descent/horizontal speeds

5. **Engine Correction** (`enable_engine_correction`): Provides guidance on appropriate engine usage

6. **Potential Guidance** (`enable_potential_guidance`): Potential-based shaping for approach to landing pad

7. **Horizontal Precision** (`enable_horizontal_precision`): Enhanced horizontal positioning guidance

**IMPORTANT**: The reward shaper maintains episode-specific state and MUST be reset at the start of each episode via `reset()` method.

**Note from CLAUDE.local.md**: A `fuel_compensation` logic exists in the `shape_reward` method to provide small reward offset when engines are used, counteracting the built-in fuel penalty to prevent over-penalization of necessary engine usage.

### Linear Controller Design

The GA approach uses a simple, interpretable linear mapping:
```
action_scores = state @ weights + bias
action = argmax(action_scores)
```

Where:
- State: 8-dimensional (x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact)
- Weights: 8×4 matrix
- Bias: 4-dimensional vector
- Actions: [0=nothing, 1=left engine, 2=main engine, 3=right engine]

This simplicity enables:
- Direct inspection of learned weights
- Understanding which state features drive which actions
- Fast evaluation and parallel training

### Model Saving Logic

DQN uses sophisticated model evaluation logic (`improved_model_evaluation_logic` in `train.py:16`):
- Primary: Landing rate improvement (most important)
- Secondary: Score improvement with same landing rate
- Special cases: Early training flexibility, milestone saves, perfect performance
- Saves as "best" (true improvement) or "improving" (promising checkpoint)

## Dependencies

```bash
# Core requirements
pip install gymnasium[classic_control]==0.29.1
pip install torch==2.1.0
pip install numpy==1.24.3
pip install matplotlib==3.7.2

# For full environment support (Box2D physics)
pip install gymnasium[box2d]

# Optional: For optimal CMA-ES performance
pip install cma
# If not installed, simple GA fallback is used automatically
```

## Directory Structure

- `ga_models/`: Saved GA models and training history (gitignored)
- `videos/`: Generated episode videos (gitignored)
- `old/`: Archived code (gitignored)
- `publish/`: Published visualizations (e.g., dqn-visualization.html)
- `tests/`: Debug and diagnostic tools (debug_analyzer.py, speed_diagnostics.py)

## Development Notes

### When modifying reward shaping:
1. Understand the 7 different reward components and their flags
2. Test with ablation studies by disabling specific components
3. Always call `reward_shaper.reset()` at episode start
4. Be aware of the fuel_compensation logic to avoid over-penalizing engine use

### When working with GA:
1. CMA-ES requires `cma` package for best performance, but has fallback
2. Parallel evaluation (`--parallel N`) provides near-linear speedup on multi-core CPUs
3. Fitness function combines reward and landing success: `fitness = avg_reward + (landing_rate × 100)`
4. Use `--describe` flag to interpret learned controller weights

### When debugging training:
1. Use `debug_model_saving()` in train.py:81 to understand why models are/aren't saved
2. Check logger.py for training diagnostics
3. GA training history saved to JSON for easy plotting
4. Tests work without Box2D by using mock environments
