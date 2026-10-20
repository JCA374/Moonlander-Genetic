"""
Lesson Video Generator

Renders the narrated lesson videos for the Lessons tab of the web page:
docs/lessons/lesson-NN.mp4, .vtt (captions), .jpg (poster) and lessons.json (titles + transcripts).
Narration for lesson N comes from voice/lesson-NN/ (voice-text pipeline, see README.md).

Usage (from the repository root, after training):
    python video_tools/make_lessons.py              # all lessons
    python video_tools/make_lessons.py --only 5 7   # some of them
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes as S  # noqa: E402
import timeline as T  # noqa: E402
import videos_def as V  # noqa: E402

# poster frame per lesson: (scene index, beat, fraction of the beat); default (1, 0, 0.7)
POSTERS = {1: (1, 2, 0.9), 3: (1, 1, 0.95), 4: (3, 2, 0.5), 5: (2, 2, 0.97), 6: (1, 1, 0.97),
           7: (1, 0, 0.95)}


def main():
    ap = argparse.ArgumentParser(description='Render the narrated lesson videos.')
    ap.add_argument('--models', default=os.path.join(S.REPO_ROOT, 'ga_models'))
    ap.add_argument('--voice-root', default=os.path.join(S.REPO_ROOT, 'voice'))
    ap.add_argument('--out', default=os.path.join(S.REPO_ROOT, 'docs', 'lessons'))
    ap.add_argument('--only', type=int, nargs='*')
    ap.add_argument('--crf', type=int, default=30)
    ap.add_argument('--posters-only', action='store_true')
    args = ap.parse_args()

    ctx = S.Context(args.models, os.path.join(S.REPO_ROOT, 'videos', 'cache'))
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'lessons.json')
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = {x['slug']: x for x in json.load(fh)}
    for n in args.only or range(1, len(V.LESSONS) + 1):
        slug, scenes = V.lesson(ctx, n)
        voice = os.path.join(args.voice_root, slug)
        out = os.path.join(args.out, f'{slug}.mp4')
        si, beat, frac = POSTERS.get(n, (1, 0, 0.7))
        T.still(scenes[si], beat, frac, voice).save(os.path.join(args.out, f'{slug}.jpg'), quality=84)
        if args.posters_only:
            continue
        timeline = T.render(scenes, out, voice, crf=args.crf)
        _, title, sub = V.LESSONS[n - 1]
        index[slug] = {'slug': slug, 'n': n, 'title': title, 'subtitle': sub,
                       'seconds': round(sum(b.end for _, b, _ in timeline), 1),
                       'transcript': T.transcript(timeline)}
    if args.posters_only:
        return
    with open(index_path, 'w') as fh:
        json.dump([index[k] for k in sorted(index)], fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
