# File Organization

This document describes the purpose of each file in the repository.

## Core Training Files

- **train_ga.py** - Main GA training script using CMA-ES
- **linear_controller.py** - Simple linear policy controller (36 params)
- **ga_evaluator.py** - Fitness evaluation for GA controllers
- **cma_es_optimizer.py** - CMA-ES optimizer with simple GA fallback
- **play_ga.py** - Visualize trained GA controllers
- **plot_ga_training.py** - Plot GA training progress
- **README_GA.md** - Complete GA documentation

## Video Generation

- **videos/ga_progress_video.py** - Create training progression videos from GA checkpoints
- **videos/merge_movies.py** - Merge multiple videos with custom text overlays
- **videos/README.md** - Video generation documentation

## Test Files

- **test_ga_components.py** - Unit tests for GA components (works without Box2D)
- **test_integration.py** - Integration test for full GA pipeline

## Configuration Files

- **requirements.txt** - Python package dependencies
- **.gitignore** - Git ignore patterns
- **CLAUDE.md** - Claude Code guidance for this repository
- **CLAUDE.local.md** - Local project notes (not in git)

## Directories

- **old/** - Archived old versions (in .gitignore)
- **videos/** - Generated videos (in .gitignore)
- **publish/** - Published visualizations
- **ga_models/** - Saved GA models (in .gitignore)
- **tests/** - Test utilities
- **__pycache__/** - Python cache (in .gitignore)

## Models and Outputs (Not in Git)

- **ga_models/ga_best.npy** - Best GA model
- **ga_models/ga_checkpoint_*.npy** - GA checkpoints (saved every 20 generations)
- **ga_models/training_history.json** - GA training history

## Quick Reference

### To train:
```bash
# Standard training (~20-40 minutes)
python train_ga.py

# Quick test
python train_ga.py --generations 50 --population 30 --episodes 5

# With parallelization
python train_ga.py --parallel 4
```

### To visualize trained models:
```bash
# Play episodes
python play_ga.py --model ga_models/ga_best.npy --episodes 5

# Describe what was learned
python play_ga.py --model ga_models/ga_best.npy --describe

# Plot training progress
python plot_ga_training.py
```

### To run tests:
```bash
python test_ga_components.py         # GA component tests
python test_integration.py           # Integration test
```

### To create progression videos:
```bash
# Create video showing training evolution
cd videos
python ga_progress_video.py

# Output: videos/ga_progress/ga_progression_YYYYMMDD_HHMMSS.mp4
```
