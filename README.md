# Moonlander-Genetic

**Landing on the Moon with 36 numbers.** This project evolves a tiny linear controller for
Gymnasium's [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) using
**CMA-ES**, an evolution strategy. There are no neural networks and no gradients, only a
population of candidate controllers that gets better every generation.

▶ **Narrated explainer, 7 short video lessons, and project page:** https://jca374.github.io/Moonlander-Genetic/

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # gymnasium[box2d], numpy, matplotlib, opencv, cma

python train_ga.py --parallel 4          # 200 generations; ~10–20 minutes on a multi-core CPU
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

### 3. Landing sites

Every episode generates a new moon: the mountains are random and the lander gets a random push at the
start. The pad itself is always in the middle at the same height, and the lander's x and y inputs are
measured from the pad, so "go to (0, 0)" is the same goal everywhere. Each pilot is scored on 10 random
sites, so evolution has to find a general rule, not memorise one flight.

### 4. Fitness

The default `balanced` mode:

```
fitness = avg_reward + 100 × landing_rate + speed_bonus × landing_rate − 0.5 × settle_frames
speed_bonus = 30 × (1 − avg_episode_length / 1000)
```

`settle_frames` is the average number of frames per episode in which an engine fires while both legs are
already on the ground. Without it, evolution happily keeps pulsing the (nearly free) side engines after
touchdown. With it, the pilot learns to land and then stay still.

`avg_reward` is the environment's own reward. It already penalises fuel use, crashes, and
drifting away from the pad. The landing term rewards actually touching down, and the speed bonus
favours landings that come quickly but only counts when the lander lands.

| `--fitness-mode` | Formula | Favours |
|---|---|---|
| `standard` | `reward + 100·landing` | the original objective |
| `balanced` (default) | `reward + 100·landing + 30·speed·landing − 0.5·settle` | safe, quick landings that stay still |
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

Outputs go to `ga_models/`: `ga_best.npy`, `ga_checkpoint_XXXX.npy`, `training_history.json`, and
`population_log.npz` (every candidate, its fitness and landing rate, and the CMA-ES mean and σ for each
generation, used by the generation visualizations; turn it off with `--no-log-population`).

## Repository layout

```
linear_controller.py     36-parameter linear policy
ga_evaluator.py          fitness evaluation (+ parallel evaluation)
cma_es_optimizer.py      CMA-ES wrapper with simple-GA fallback
train_ga.py              training loop, checkpoints, history
play_ga.py               watch / evaluate / describe a controller
plot_ga_training.py      training curves
tests/                   unit + integration tests (mock env, no Box2D needed)
video_tools/             explainer + lesson video generators, progression video
voice/                   narration work dirs (voice-text pipeline; WAVs are gitignored)
docs/                    GitHub Pages site (index.html, explainer, lessons/)
```

## Tests

```bash
python tests/test_ga_components.py
python tests/test_integration.py
```

## Videos

```bash
python video_tools/ga_progress_video.py   # gameplay from each checkpoint → videos/ga_progress/
python video_tools/make_explainer.py      # narrated explainer → docs/explainer.mp4 (+ .vtt, stills in docs/img/)
python video_tools/make_lessons.py        # lessons → docs/lessons/lesson-NN.mp4 (+ .vtt, .jpg, lessons.json)
```

Scenes live in `video_tools/scenes.py`, and the scene list of each video is in `video_tools/videos_def.py`.
The narration scripts are `video_tools/narration/*.md`. Each `## <scene>` or `## <scene>.<n>` section is one
*beat* of a scene, and the video stretches each beat to the length of its narration (`video_tools/timeline.py`).

Narration is voiced locally with Kokoro TTS through the
`voice-text` pipeline from [fantasy-audiobook-studio](https://github.com/JCA374/fantasy-audiobook-studio):

```bash
V=~/.claude/skills/voice-text/scripts/voice.sh
$V new video_tools/narration/explainer.md --kind voiceover --slug explainer   # → voice/explainer/
$V scan voice/explainer      # speech-flow check
$V prep voice/explainer && $V narrate voice/explainer
$V listen voice/explainer    # Whisper listens back for skipped or garbled words
$V export voice/explainer    # per-beat MP3s + timings.json used by the renderer
$V status voice/explainer    # exit 0 = every audio stage passed
```

After editing a script, rebuild its work dir with `$V new ... --force`, then repeat the steps. Without
narration the videos render silent with default timings, which is handy for previews.

## References

- N. Hansen (2016), *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772
- Gymnasium LunarLander-v2 (Farama Foundation)
