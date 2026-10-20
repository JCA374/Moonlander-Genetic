"""
Explainer Video Generator

Renders the narrated explainer (docs/explainer.mp4 + .vtt captions) and the still images used by
the web page. Narration comes from voice/explainer/ (voice-text pipeline, see README.md); without
it the video renders silent with default timings, which is handy for previews.

Usage (from the repository root, after training):
    python video_tools/make_explainer.py
    python video_tools/make_explainer.py --stills-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes as S  # noqa: E402
import timeline as T  # noqa: E402
import videos_def as V  # noqa: E402

# still name -> (scene name, beat, fraction of the beat)
STILLS = {'poster': ('title', 0, 0.5), 'controller': ('controller', 1, 0.6),
          'population': ('population', 2, 0.8), 'cmaes': ('cmaes', 2, 0.3),
          'training': ('gen_fitness', 2, 0.95), 'ghosts': ('gen_ghosts', 2, 0.97),
          'cloud': ('gen_cloud', 2, 0.9), 'evolution': ('evolution', 1, 0.97),
          'worked': ('w_worked', 0, 0.9), 'policy': ('w_policy_map', 1, 0.95),
          'weights': ('w_story', 2, 0.9)}


def main():
    ap = argparse.ArgumentParser(description='Render the narrated explainer video.')
    ap.add_argument('--models', default=os.path.join(S.REPO_ROOT, 'ga_models'))
    ap.add_argument('--voice', default=os.path.join(S.REPO_ROOT, 'voice', 'explainer'))
    ap.add_argument('--output', default=os.path.join(S.REPO_ROOT, 'docs', 'explainer.mp4'))
    ap.add_argument('--stills', default=os.path.join(S.REPO_ROOT, 'docs', 'img'))
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--stills-only', action='store_true')
    args = ap.parse_args()

    ctx = S.Context(args.models, os.path.join(S.REPO_ROOT, 'videos', 'cache'))
    print(f'Best controller: {ctx.stats}')
    scenes = V.explainer(ctx)
    by_name = {s.name: s for s in scenes}
    os.makedirs(args.stills, exist_ok=True)
    for still, (scene, beat, frac) in STILLS.items():
        T.still(by_name[scene], beat, frac, args.voice).save(os.path.join(args.stills, f'{still}.jpg'),
                                                             quality=86)
    with open(os.path.join(args.stills, 'stats.json'), 'w') as fh:
        json.dump({**ctx.stats, 'generations': len(ctx.history['generation']),
                   'best_fitness': float(max(ctx.history['best_fitness']))}, fh, indent=2)
    if args.stills_only:
        return
    T.render(scenes, args.output, args.voice, crf=args.crf)


if __name__ == '__main__':
    main()
