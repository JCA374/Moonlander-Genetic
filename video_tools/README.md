# Video Generation Scripts

This directory contains tools for creating videos showing GA training progression.

## GA Progress Video Generator

Creates a comprehensive video showing the evolution of your genetic algorithm training.

### Usage

```bash
# From the repository root
python video_tools/ga_progress_video.py
```

This will:
1. Find all GA checkpoints in `ga_models/`
2. Load training history to get fitness and landing rates
3. Record episodes with each checkpoint model
4. Create individual videos with overlays showing generation, fitness, and landing rate
5. Merge them into a single progression video

### Configuration

Edit `ga_progress_video.py` to customize:

```python
maker = GAProgressVideoMaker(
    ga_models_dir='../ga_models',     # Where to find GA models
    output_dir='ga_progress',          # Where to save videos
    fps=30,                            # Frames per second
    max_episode_steps=1000,            # Max steps per episode
    episodes_per_checkpoint=3          # Record N episodes, use best
)
```

### Output

- Individual checkpoint videos: `videos/ga_progress/XXXX_ga_checkpoint_XXXX.mp4`
- Merged progression video: `videos/ga_progress/ga_progression_YYYYMMDD_HHMMSS.mp4`

## Video Merger

Merges multiple video files with custom text overlays.

### Usage

```python
from merge_movies import MovieMerger

merger = MovieMerger(
    videos_folder='path/to/videos',
    output_name='merged_output.mp4',
    comments=[
        "Generation 20: Early learning",
        "Generation 40: Improving",
        "Generation 60: Advanced",
        "Best Model: Mastery"
    ]
)
merger.merge_videos()
```

## Requirements

```bash
pip install opencv-python numpy gymnasium
```

## Tips

1. **Reduce video size**: Lower `fps` or `max_episode_steps`
2. **Better representation**: Increase `episodes_per_checkpoint` to 5-10
3. **Faster processing**: Reduce `episodes_per_checkpoint` to 1
4. **Custom overlays**: Edit `create_overlay()` method in `ga_progress_video.py`
