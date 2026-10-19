# Moonlander-Genetic

**Landing on the Moon with 36 numbers.** This project evolves a tiny linear controller for
Gymnasium's [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) using
**CMA-ES**, an evolution strategy. There are no neural networks and no gradients, only a
population of candidate controllers that gets better every generation.

▶ **Explainer video and project page:** https://jca374.github.io/Moonlander-Genetic/

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # gymnasium[box2d], numpy, matplotlib, opencv, cma

python train_ga.py --parallel 4          # 200 generations; a few minutes on a multi-core CPU
python play_ga.py --model ga_models/ga_best.npy --episodes 5        # watch it fly
python play_ga.py --model ga_models/ga_best.npy --describe          # what did it learn?
python play_ga.py --model ga_models/ga_best.npy --evaluate 100 --no-render
python plot_ga_training.py --history ga_models/training_history.json
```

## How it works

### 1. The controller: 36 parameters

```
action_scores = state @ W + b        # W: 8×4, b: 4  →  36 parameters
action        = argmax(action_scores)
```

| State (8 inputs) | Actions (4 outputs) |
|---|---|
| x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact | 0 = do nothing, 1 = left engine, 2 = main engine, 3 = right engine |

Every state feature "votes" for every action through one weight. The action with the highest
total score fires. Since the whole policy is one 8×4 matrix, you can read off what it learned
(`--describe`).

### 2. The genetic algorithm: CMA-ES

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) keeps a **search distribution** over
the 36-dimensional parameter space, a multivariate Gaussian with a mean, a step size σ and a
covariance matrix. Each generation it:

1. **Samples** a population of controllers (default 50) from the Gaussian.
2. **Evaluates** each one by flying several episodes and computing a fitness score.
3. **Selects** the best half and moves the mean toward them, weighted by rank.
4. **Adapts** the covariance, stretching the distribution along directions that kept producing
   improvements, and adapts σ (growing it while progress is steady, shrinking it near an optimum).
5. **Repeats.**

Unlike a classic GA with crossover and bit-flip mutation, CMA-ES learns the *shape* of the
fitness landscape, which makes it very efficient on small continuous problems like this one.
If the `cma` package is not installed, `cma_es_optimizer.py` falls back to a simple
elitist GA (selection, crossover, Gaussian mutation).

### 3. Fitness

The default `balanced` mode:

```
fitness = avg_reward + 100 × landing_rate + speed_bonus × landing_rate
speed_bonus = 30 × (1 − avg_episode_length / 1000)
```

`avg_reward` is the environment's own reward. It already penalises fuel use, crashes, and
drifting away from the pad. The landing term rewards actually touching down, and the speed bonus
favours landings that come quickly but only counts when the lander lands.

| `--fitness-mode` | Formula | Favours |
|---|---|---|
| `standard` | `reward + 100·landing` | the original objective |
| `balanced` (default) | `reward + 100·landing + 30·speed·landing` | safe, quick, efficient landings |
| `speed` | `reward + 150·landing + 50·speed·landing` | fastest landings |
| `fuel` | `1.5·reward + 100·landing` | fuel efficiency |

## Training options

| Flag | Default | Meaning |
|---|---|---|
| `--generations` | 200 | evolution cycles |
| `--population` | 50 | controllers per generation |
| `--episodes` | 10 | episodes per fitness evaluation |
| `--sigma` | 0.5 | initial CMA-ES step size |
| `--parallel` | 1 | worker processes for evaluation |
| `--fitness-mode` | balanced | see the table above |
| `--checkpoint-interval` | 20 | generations between checkpoints |

If training stalls, raise `--sigma` (1.0), `--population` (100) or `--episodes` (20).
For faster runs, lower `--episodes` and use `--parallel`.

Outputs go to `ga_models/`: `ga_best.npy`, `ga_checkpoint_XXXX.npy`, and `training_history.json`.

## Repository layout

```
linear_controller.py     36-parameter linear policy
ga_evaluator.py          fitness evaluation (+ parallel evaluation)
cma_es_optimizer.py      CMA-ES wrapper with simple-GA fallback
train_ga.py              training loop, checkpoints, history
play_ga.py               watch / evaluate / describe a controller
plot_ga_training.py      training curves
tests/                   unit + integration tests (mock env, no Box2D needed)
video_tools/             progression video + explainer video generators
docs/                    GitHub Pages site (index.html + explainer video)
```

## Tests

```bash
python tests/test_ga_components.py
python tests/test_integration.py
```

## Videos

```bash
python video_tools/ga_progress_video.py   # gameplay from each checkpoint → videos/ga_progress/
python video_tools/make_explainer.py      # the captioned explainer → docs/explainer.mp4
```

## References

- N. Hansen (2016), *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772
- Gymnasium LunarLander-v2 (Farama Foundation)
