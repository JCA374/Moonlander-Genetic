"""
GA Progress Video Generator

Creates videos showing the progression of the genetic algorithm training
by recording episodes at different checkpoints and merging them into a
comprehensive training progression video.
"""

import os
import json
import glob
import re
import numpy as np
import gymnasium as gym
import cv2
from datetime import datetime
import sys

# Add parent directory to path to import GA modules
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from linear_controller import LinearController


class GAProgressVideoMaker:
    """Creates progression videos for GA training."""

    def __init__(
        self,
        ga_models_dir='ga_models',
        output_dir='videos/ga_progress',
        fps=30,
        max_episode_steps=1000,
        episodes_per_checkpoint=3
    ):
        self.ga_models_dir = ga_models_dir
        self.output_dir = output_dir
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        self.episodes_per_checkpoint = episodes_per_checkpoint

        os.makedirs(output_dir, exist_ok=True)

        # Load training history for fitness/landing rate info
        self.training_history = self.load_training_history()

    def load_training_history(self):
        """Load training history to get fitness and landing rate info."""
        history_path = os.path.join(self.ga_models_dir, 'training_history.json')

        if not os.path.exists(history_path):
            print(f"⚠️  Training history not found at {history_path}")
            return None

        with open(history_path, 'r') as f:
            history = json.load(f)

        print(f"✓ Loaded training history with {len(history['generation'])} generations")
        return history

    def discover_checkpoints(self):
        """Find all checkpoint models and sort chronologically."""
        checkpoint_files = glob.glob(os.path.join(self.ga_models_dir, 'ga_checkpoint_*.npy'))
        checkpoint_files.extend(glob.glob(os.path.join(self.ga_models_dir, 'ga_best.npy')))

        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {self.ga_models_dir}")
            return []

        checkpoints = []

        for file_path in checkpoint_files:
            filename = os.path.basename(file_path)

            if filename == 'ga_best.npy':
                generation = 999999  # Ensure best comes last
                desc = "Best Model"
            else:
                match = re.search(r'checkpoint_(\d+)', filename)
                if match:
                    generation = int(match.group(1))
                    desc = f"Generation {generation}"
                else:
                    continue

            # Get fitness and landing rate from history
            fitness = None
            landing_rate = None

            if self.training_history and generation < 999999:
                try:
                    gen_idx = self.training_history['generation'].index(generation)
                    fitness = self.training_history['best_fitness'][gen_idx]
                    landing_rate = self.training_history['best_landing_rate'][gen_idx]
                except (ValueError, IndexError):
                    pass
            elif self.training_history and generation == 999999:
                # For best model, use the last (highest) values
                fitness = self.training_history['best_fitness'][-1]
                landing_rate = self.training_history['best_landing_rate'][-1]

            checkpoints.append({
                'filename': filename,
                'path': file_path,
                'generation': generation,
                'desc': desc,
                'fitness': fitness,
                'landing_rate': landing_rate
            })

        # Sort by generation
        checkpoints.sort(key=lambda x: x['generation'])

        print(f"\n📋 Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            fitness_str = f"{cp['fitness']:.1f}" if cp['fitness'] is not None else "N/A"
            landing_str = f"{cp['landing_rate']*100:.1f}%" if cp['landing_rate'] is not None else "N/A"
            print(f"  {cp['desc']:20} - Fitness: {fitness_str:8} Landing: {landing_str}")

        return checkpoints

    def load_controller(self, model_path):
        """Load a GA controller from file."""
        try:
            params = np.load(model_path)
            controller = LinearController()
            controller.set_parameters(params)
            return controller
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            return None

    def record_episode(self, controller, seed):
        """Record a single episode with the given controller."""
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        state, _ = env.reset(seed=seed)

        frames = []
        total_reward = 0
        done = False
        step = 0

        while not done and step < self.max_episode_steps:
            # Capture frame
            frame = env.render()
            frames.append(frame)

            # Get action from controller
            action = controller.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state
            step += 1

        env.close()

        # Determine success
        success = terminated and total_reward > 0

        return {
            'frames': frames,
            'total_reward': total_reward,
            'steps': len(frames),
            'success': success
        }

    def create_overlay(self, frame, checkpoint, episode_data, overlay_height=120):
        """Add informative overlay to frame showing training progress."""
        h, w = frame.shape[:2]

        # Create extended canvas with overlay space at bottom
        canvas = np.zeros((h + overlay_height, w, 3), dtype=np.uint8)
        canvas[:h, :] = frame

        # Overlay background
        overlay_bg = (20, 20, 20)
        canvas[h:, :] = overlay_bg

        # Draw separator line
        cv2.line(canvas, (0, h), (w, h), (80, 80, 80), 2)

        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Line 1: Generation
        text1 = checkpoint['desc']
        font_scale1 = 1.2
        thickness1 = 2
        text_color = (255, 255, 255)

        text_size1 = cv2.getTextSize(text1, font, font_scale1, thickness1)[0]
        text_x1 = (w - text_size1[0]) // 2
        text_y1 = h + 35

        cv2.putText(canvas, text1, (text_x1, text_y1), font,
                   font_scale1, text_color, thickness1, cv2.LINE_AA)

        # Line 2: Fitness and Landing Rate
        if checkpoint['fitness'] is not None and checkpoint['landing_rate'] is not None:
            text2 = f"Fitness: {checkpoint['fitness']:.1f}  |  Landing Rate: {checkpoint['landing_rate']*100:.1f}%"
        else:
            text2 = f"Reward: {episode_data['total_reward']:.1f}  |  {'SUCCESS' if episode_data['success'] else 'FAILED'}"

        font_scale2 = 0.8
        thickness2 = 2

        text_size2 = cv2.getTextSize(text2, font, font_scale2, thickness2)[0]
        text_x2 = (w - text_size2[0]) // 2
        text_y2 = h + 75

        cv2.putText(canvas, text2, (text_x2, text_y2), font,
                   font_scale2, text_color, thickness2, cv2.LINE_AA)

        return canvas

    def create_checkpoint_video(self, checkpoint, seed_offset=42):
        """Create video for a single checkpoint."""
        print(f"\n🎥 Recording {checkpoint['desc']}...")

        # Load controller
        controller = self.load_controller(checkpoint['path'])
        if controller is None:
            return None

        # Record multiple episodes and pick the best one
        best_episode = None
        best_reward = float('-inf')

        for i in range(self.episodes_per_checkpoint):
            episode_data = self.record_episode(controller, seed_offset + i * 100)

            if episode_data['total_reward'] > best_reward:
                best_reward = episode_data['total_reward']
                best_episode = episode_data

        if best_episode is None:
            print(f"   ❌ Failed to record episodes")
            return None

        print(f"   📊 Best episode: {best_episode['total_reward']:.1f} points, "
              f"{best_episode['steps']} steps, "
              f"{'SUCCESS' if best_episode['success'] else 'FAILED'}")

        # Create video file
        video_filename = f"{checkpoint['generation']:04d}_{checkpoint['filename'].replace('.npy', '')}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)

        # Get frame dimensions
        first_frame = best_episode['frames'][0]
        h, w = first_frame.shape[:2]
        overlay_height = 120

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h + overlay_height))

        # Write frames with overlay
        for frame in best_episode['frames']:
            frame_with_overlay = self.create_overlay(frame, checkpoint, best_episode, overlay_height)
            writer.write(frame_with_overlay)

        # Hold last frame for 1 second
        for _ in range(self.fps):
            writer.write(frame_with_overlay)

        writer.release()

        print(f"   ✅ Video saved: {video_filename}")

        return {
            'video_path': video_path,
            'checkpoint': checkpoint,
            'episode_data': best_episode
        }

    def merge_videos(self):
        """Merge individual checkpoint videos into progression video."""
        print(f"\n🎬 Merging checkpoint videos...")

        # Find all checkpoint videos
        video_files = sorted(glob.glob(os.path.join(self.output_dir, '*.mp4')))

        if not video_files:
            print("❌ No checkpoint videos found to merge")
            return None

        print(f"   Found {len(video_files)} videos to merge")

        # Read first video to get dimensions
        cap = cv2.VideoCapture(video_files[0])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create merged video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.output_dir, f'ga_progression_{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Concatenate all videos
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()

        writer.release()

        print(f"✅ Merged video saved: {output_path}")
        return output_path

    def create_progression_video(self, seed_offset=42):
        """Create complete progression video showing training evolution."""
        print("="*70)
        print("GA TRAINING PROGRESSION VIDEO MAKER")
        print("="*70)

        # Discover checkpoints
        checkpoints = self.discover_checkpoints()

        if not checkpoints:
            print("❌ No checkpoints found")
            return None

        # Create individual videos for each checkpoint
        video_info = []
        for checkpoint in checkpoints:
            result = self.create_checkpoint_video(checkpoint, seed_offset)
            if result:
                video_info.append(result)

        if not video_info:
            print("❌ No videos were created successfully")
            return None

        # Merge all videos
        merged_video = self.merge_videos()

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Individual videos: {len(video_info)}")
        print(f"Merged video: {merged_video}")
        print(f"Output directory: {self.output_dir}")

        return merged_video


def main():
    """Main function to create GA progression video."""
    maker = GAProgressVideoMaker(
        ga_models_dir=os.path.join(REPO_ROOT, 'ga_models'),
        output_dir=os.path.join(REPO_ROOT, 'videos', 'ga_progress'),
        fps=30,
        max_episode_steps=1000,
        episodes_per_checkpoint=3
    )

    video_path = maker.create_progression_video(seed_offset=42)

    if video_path:
        print(f"\n🎉 Video creation complete!")
        print(f"📹 Watch your GA training progress: {video_path}")
    else:
        print(f"\n❌ Video creation failed")


if __name__ == '__main__':
    main()
