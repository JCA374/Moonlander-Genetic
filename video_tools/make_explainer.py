"""
Explainer Video Generator

Renders a captioned explainer video (no voice-over) showing how the project works:
the task, the 36-parameter linear controller, how CMA-ES evolves it, the fitness
function, training progress, gameplay across generations and what was learned.

Frames are drawn with Pillow (gameplay comes from gymnasium's rgb_array renderer)
and piped straight into ffmpeg.

Usage (from the repository root, after training with checkpoints):
    python video_tools/make_explainer.py
    python video_tools/make_explainer.py --output docs/explainer.mp4 --stills docs/img
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import textwrap

import cma
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from linear_controller import LinearController  # noqa: E402

W, H, FPS = 1280, 720, 30

BG = (11, 16, 32)
PANEL = (22, 30, 54)
FG = (232, 236, 245)
MUTED = (140, 152, 180)
AMBER = (255, 184, 64)
BLUE = (96, 168, 255)
GREEN = (84, 212, 140)
RED = (240, 96, 96)

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
SHOW_CAPTIONS = True
_font_cache = {}

STATE_NAMES = ['x position', 'y position (altitude)', 'horizontal velocity', 'vertical velocity',
               'angle', 'angular velocity', 'left leg contact', 'right leg contact']
STATE_SHORT = ['x', 'y', 'vx', 'vy', 'angle', 'ang. vel', 'left leg', 'right leg']
ACTION_NAMES = ['do nothing', 'left engine', 'main engine', 'right engine']
ACTION_SHORT = ['none', 'left', 'main', 'right']
# What a positive / negative value of each input means physically
STATE_MEANING = [
    ('right of the pad', 'left of the pad'),
    ('high up', 'low'),
    ('drifting right', 'drifting left'),
    ('rising', 'falling'),
    ('tilted left', 'tilted right'),
    ('rotating left', 'rotating right'),
    ('touching down with the left leg', None),
    ('touching down with the right leg', None),
]


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------

def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _font_cache:
        name = 'DejaVuSansMono' if mono else 'DejaVuSans'
        if bold:
            name += '-Bold'
        _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name + '.ttf'), size)
    return _font_cache[key]


def mix(c1, c2, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, start, dur=0.4):
    return ease((t - start) / dur)


def blank():
    return Image.new('RGB', (W, H), BG)


def text(d, xy, s, size=24, fill=FG, a=1.0, bold=False, mono=False, anchor='la', bg=BG):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, bold, mono), fill=mix(bg, fill, a), anchor=anchor)


def heading(d, title, t, sub=None):
    a = fade(t, 0)
    text(d, (60, 42), title, 38, FG, a, bold=True)
    d.rectangle([60, 92, 60 + int(120 * a), 96], fill=AMBER)
    if sub:
        text(d, (60 + 140, 84), sub, 18, MUTED, a)


def caption(img, captions, t):
    """Draw the active caption (list of (start, end, text)) in a bar at the bottom."""
    if not SHOW_CAPTIONS:
        return
    for start, end, s in captions:
        if start <= t < end:
            a = min(fade(t, start, 0.3), 1 - fade(t, end - 0.3, 0.3))
            overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            lines = textwrap.wrap(s, 78)
            bar_h = 26 + 34 * len(lines)
            od.rounded_rectangle([80, H - bar_h - 18, W - 80, H - 18], 14,
                                 fill=(6, 9, 20, int(225 * a)))
            for i, line in enumerate(lines):
                od.text((W // 2, H - bar_h - 18 + 14 + 34 * i), line, font=font(26),
                        fill=FG + (int(255 * a),), anchor='ma')
            img.paste(Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'))
            return


def paste_frame(img, frame, box, border=None, dim=0.0):
    x, y, w, h = box
    tile = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    if dim > 0:
        tile = Image.blend(tile, Image.new('RGB', (w, h), BG), dim)
    img.paste(tile, (x, y))
    if border:
        ImageDraw.Draw(img).rectangle([x - 3, y - 3, x + w + 2, y + h + 2], outline=border, width=3)


def weight_color(v, vmax):
    a = min(1.0, abs(v) / vmax)
    return mix(PANEL, AMBER if v > 0 else BLUE, a)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_controller(path):
    c = LinearController()
    c.set_parameters(np.load(path))
    return c


def record(controller, seed, max_steps=1000, frames=True):
    env = gym.make('LunarLander-v2', render_mode='rgb_array' if frames else None)
    state, _ = env.reset(seed=seed)
    out = {'frames': [], 'states': [], 'actions': [], 'total': 0.0}
    terminated = False
    reward = 0.0
    for _ in range(max_steps):
        if frames:
            out['frames'].append(env.render())
        action = controller.get_action(state)
        out['states'].append(state.copy())
        out['actions'].append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        out['total'] += reward
        if terminated or truncated:
            break
    if frames:
        out['frames'].append(env.render())
    env.close()
    out['landed'] = bool(terminated and reward > 0)
    out['steps'] = len(out['actions'])
    return out


def find_checkpoints(models_dir):
    cps = {}
    for p in glob.glob(os.path.join(models_dir, 'ga_checkpoint_*.npy')):
        cps[int(re.search(r'(\d+)\.npy$', p).group(1))] = p
    return dict(sorted(cps.items()))


# ----------------------------------------------------------------------------
# Scenes - each is (duration_seconds, render(t) -> PIL.Image)
# ----------------------------------------------------------------------------

def scene_title(best_ep):
    caps = [(1.2, 7.0, 'A genetic algorithm teaches a lunar lander to land, '
                       'using a controller made of only 36 numbers.')]
    frames = best_ep['frames']

    def render(t):
        img = blank()
        f = frames[min(int(t * FPS), len(frames) - 1)]
        tile = Image.fromarray(f).resize((1280, 853), Image.BILINEAR).crop((0, 133, 1280, 853))
        tile = Image.blend(tile, Image.new('RGB', tile.size, BG), 0.62)
        img.paste(tile, (0, 0))
        d = ImageDraw.Draw(img)
        a = fade(t, 0.2, 0.8)
        text(d, (W // 2, 230), 'Moonlander-Genetic', 72, FG, a, bold=True, anchor='mm')
        text(d, (W // 2, 310), 'Evolving a 36-parameter lunar lander pilot with CMA-ES', 30, AMBER,
             fade(t, 0.7, 0.8), anchor='mm')
        caption(img, caps, t)
        return img
    return 7.5, render


def scene_task(best_ep):
    caps = [(0.3, 4.6, 'The task: land softly between the two flags. Gravity pulls the lander down; '
                       'engines cost fuel.'),
            (4.6, 9.0, 'Every step the lander observes 8 numbers about itself...'),
            (9.0, 13.5, '...and chooses one of 4 actions. That is the whole interface.')]
    frames = best_ep['frames']

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The problem', t, 'Gymnasium LunarLander-v2')
        paste_frame(img, frames[min(int(t * FPS), len(frames) - 1)], (60, 130, 660, 440))
        x0 = 770
        a1 = fade(t, 4.6)
        text(d, (x0, 130), '8 inputs (state)', 26, BLUE, a1, bold=True)
        for i, name in enumerate(STATE_NAMES):
            text(d, (x0 + 10, 172 + 28 * i), f'{i + 1}. {name}', 21, FG, fade(t, 4.8 + 0.2 * i))
        a2 = fade(t, 9.0)
        text(d, (x0, 420), '4 outputs (actions)', 26, AMBER, a2, bold=True)
        for i, name in enumerate(ACTION_NAMES):
            text(d, (x0 + 10, 462 + 28 * i), f'{i}. {name}', 21, FG, fade(t, 9.2 + 0.2 * i))
        caption(img, caps, t)
        return img
    return 13.5, render


def scene_controller(best_ctrl, best_ep):
    caps = [(0.3, 5.0, 'The pilot is a single matrix multiplication: scores = state × W + b.'),
            (5.0, 10.0, 'Each input votes for each action through one weight '
                        '(orange = positive, blue = negative).'),
            (10.0, 15.0, 'The action with the highest score fires. Watch the scores change '
                         'as the lander flies.'),
            (15.0, 19.5, '8 × 4 weights + 4 biases = 36 parameters. Finding good values is the '
                         'genetic algorithm\'s job.')]
    Wm, b = best_ctrl.weights, best_ctrl.bias
    vmax = np.abs(Wm).max()
    frames, states, actions = best_ep['frames'], best_ep['states'], best_ep['actions']
    n = len(states)

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The controller: 36 numbers', t)
        k = int(t * FPS) % (n + 45)
        k = min(k, n - 1)
        s, act = states[k], actions[k]
        paste_frame(img, frames[k], (40, 140, 510, 340))
        text(d, (295, 500), f'action: {ACTION_NAMES[act]}', 26, AMBER, bold=True, anchor='ma')

        # state bars
        x_lab, x_bar, x_hm, row0, rh = 580, 700, 850, 165, 38
        text(d, (x_lab, 112), 'state', 18, BLUE, bold=True)
        text(d, (x_hm + 4 * 47 // 2, 112), 'weights W', 18, MUTED, bold=True, anchor='ma')
        for i in range(8):
            y = row0 + i * rh
            text(d, (x_lab, y + 6), STATE_SHORT[i], 18, FG)
            d.rectangle([x_bar, y + 6, x_bar + 120, y + 26], fill=PANEL)
            v = float(np.clip(s[i], -1.5, 1.5)) / 1.5 * 60
            cx = x_bar + 60
            d.rectangle([min(cx, cx + v), y + 6, max(cx, cx + v), y + 26], fill=BLUE)
            d.line([cx, y + 3, cx, y + 29], fill=MUTED)
            for j in range(4):
                cx0 = x_hm + j * 47
                d.rectangle([cx0, y, cx0 + 44, y + rh - 4], fill=weight_color(Wm[i, j], vmax))
        for j in range(4):
            text(d, (x_hm + j * 47 + 22, row0 - 16), ACTION_SHORT[j], 14, MUTED, anchor='mm')
        yb = row0 + 8 * rh + 6
        text(d, (x_lab, yb + 4), '+ bias b', 18, MUTED)
        for j in range(4):
            cx0 = x_hm + j * 47
            d.rectangle([cx0, yb, cx0 + 44, yb + 26], fill=weight_color(b[j], vmax))

        # scores
        scores = s @ Wm + b
        xs = 1060
        text(d, (xs, 112), 'scores', 18, AMBER, bold=True)
        smax = max(1e-6, np.abs(scores).max())
        for j in range(4):
            y = row0 + j * 70
            win = j == int(np.argmax(scores))
            text(d, (xs, y), ACTION_SHORT[j], 18, AMBER if win else FG, bold=win)
            d.rectangle([xs, y + 26, xs + 130, y + 50], fill=PANEL)
            wbar = int(abs(scores[j]) / smax * 130)
            d.rectangle([xs, y + 26, xs + wbar, y + 50], fill=AMBER if win else MUTED)
            text(d, (xs + 136, y + 28), f'{scores[j]:+.1f}', 15, FG, anchor='la')
        text(d, (xs, row0 + 4 * 70 + 10), 'argmax → action', 17, MUTED)
        caption(img, caps, t)
        return img
    return 19.5, render


def scene_population(seed=7, count=18):
    rng = np.random.default_rng(seed)
    init = LinearController().get_parameters() * 0
    eps = []
    for i in range(count):
        c = LinearController()
        c.set_parameters(init + rng.standard_normal(36) * 0.5)
        eps.append(record(c, seed=100 + i, max_steps=600))
    scores = [e['total'] + (100 if e['landed'] else 0) for e in eps]
    rank = np.argsort(scores)[::-1]
    selected = set(rank[:count // 2].tolist())
    caps = [(0.3, 5.5, 'Generation 0: CMA-ES samples a population of random controllers '
                       '(50 in training, 18 shown here).'),
            (5.5, 10.5, 'Each one flies and gets a fitness score. Random pilots crash, '
                        'some just less badly than others.'),
            (10.5, 16.0, 'The best half are selected. They decide where the next generation is '
                         'sampled from.')]
    tw, th, gap = 165, 110, 16
    x0 = (W - (6 * tw + 5 * gap)) // 2
    y0 = 125

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'A population of pilots', t)
        step = int(t * FPS * 2)
        sel_a = fade(t, 10.5, 0.8)
        for i, ep in enumerate(eps):
            r, c = divmod(i, 6)
            x, y = x0 + c * (tw + gap), y0 + r * (th + 42)
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            dim = 0.65 * sel_a if i not in selected else 0.0
            border = None
            if sel_a > 0 and i in selected:
                border = mix(BG, AMBER, sel_a)
            paste_frame(img, ep['frames'][k], (x, y, tw, th), border=border, dim=dim)
            if done:
                lab = 'landed' if ep['landed'] else ('crashed' if ep['steps'] < 600 else 'timeout')
                col = GREEN if ep['landed'] else RED
                text(d, (x + 4, y + th + 8), lab, 16, col, 1 - 0.6 * (dim > 0))
                text(d, (x + tw - 4, y + th + 8), f'{scores[i]:+.0f}', 16, FG, 1 - 0.6 * (dim > 0),
                     anchor='ra')
        caption(img, caps, t)
        return img
    return 16.0, render


def scene_loop():
    steps = [('Sample', 'draw a population of 36-number\ncontrollers from a Gaussian'),
             ('Evaluate', 'fly each one for several\nepisodes, compute fitness'),
             ('Select', 'rank them, keep\nthe best half'),
             ('Update', 'move the mean toward the winners,\nreshape the covariance')]
    caps = [(0.3, 5.0, 'This loop repeats every generation: sample, evaluate, select, update.'),
            (5.0, 10.0, 'Nothing is differentiated. Only the fitness score is used, so any '
                        'controller structure works.')]
    cx, cy, R = W // 2, 370, 190
    pos = [(cx, cy - R), (cx + R + 110, cy), (cx, cy + R), (cx - R - 110, cy)]

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The evolutionary loop', t)
        active = int(t / 1.2) % 4
        d.ellipse([cx - R - 110, cy - R, cx + R + 110, cy + R], outline=PANEL, width=4)
        for i, (name, desc) in enumerate(steps):
            px, py = pos[i]
            on = i == active
            d.rounded_rectangle([px - 150, py - 48, px + 150, py + 48], 14,
                                fill=mix(PANEL, AMBER, 0.25 if on else 0), outline=AMBER if on else MUTED,
                                width=3 if on else 1)
            text(d, (px, py - 30), f'{i + 1}. {name}', 24, AMBER if on else FG, bold=True,
                 anchor='ma', bg=PANEL)
            for li, line in enumerate(desc.split('\n')):
                text(d, (px, py + 2 + 20 * li), line, 16, FG, anchor='ma', bg=PANEL)
        text(d, (cx, cy), f'generation {int(t / 4.8) + 1}', 22, MUTED, anchor='mm')
        caption(img, caps, t)
        return img
    return 10.0, render


def scene_cmaes():
    # Toy 2-D problem: a rotated, narrow valley with its minimum at (2, 1.5)
    ang = np.deg2rad(35)
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    opt = np.array([2.0, 1.5])

    def f(p):
        u = Rm.T @ (np.asarray(p) - opt)
        return u[0] ** 2 + 25 * u[1] ** 2

    es = cma.CMAEvolutionStrategy([-3.0, -2.6], 0.6, {'popsize': 12, 'seed': 3, 'verbose': -9})
    gens = []
    for g in range(40):
        mean0, cov0 = es.mean.copy(), es.sigma ** 2 * np.array(es.C)
        sols = es.ask()
        fit = [f(s) for s in sols]
        es.tell(sols, fit)
        gens.append({'sols': np.array(sols), 'fit': np.array(fit), 'mean0': mean0, 'cov0': cov0,
                     'mean1': es.mean.copy(), 'cov1': es.sigma ** 2 * np.array(es.C),
                     'sigma': es.sigma})
        if es.sigma < 0.01:
            break

    # Landscape background
    pw, ph = 700, 520
    xmin, xmax, ymin, ymax = -4.5, 4.0, -3.6, 3.0
    xs, ys = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 240))
    Z = np.vectorize(lambda a, b: f((a, b)))(xs, ys)
    fig = plt.figure(figsize=(pw / 100, ph / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(xs, ys, np.log1p(Z), levels=24, cmap='magma')
    ax.contour(xs, ys, np.log1p(Z), levels=12, colors='white', linewidths=0.4, alpha=0.35)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')
    fig.canvas.draw()
    land = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    land = Image.blend(land, Image.new('RGB', land.size, BG), 0.25)
    ox, oy = 60, 125

    def to_px(p):
        return (ox + (p[0] - xmin) / (xmax - xmin) * pw, oy + (ymax - p[1]) / (ymax - ymin) * ph)

    def ellipse_pts(mean, cov, k=1.5):
        vals, vecs = np.linalg.eigh(cov)
        th = np.linspace(0, 2 * np.pi, 64)
        pts = (vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12)) * k) @ np.vstack([np.cos(th), np.sin(th)])).T
        return [to_px(mean + p) for p in pts]

    slow, slow_dur, fast_dur = 6, 1.6, 0.45
    times = [slow_dur] * min(slow, len(gens)) + [fast_dur] * max(0, len(gens) - slow)
    starts = np.concatenate([[0.6], 0.6 + np.cumsum(times)])
    total = starts[-1] + 3.0
    caps = [(0.3, 4.8, 'How CMA-ES searches, shown in 2-D (the real problem has 36 dimensions). '
                       'Bright = high cost, dark = optimum.'),
            (4.8, 9.6, 'It keeps a Gaussian cloud of candidates. Each generation: sample (dots), '
                       'keep the best half (amber)...'),
            (9.6, 14.0, '...then move the mean toward them and reshape the covariance (the ellipse).'),
            (14.0, 19.0, 'The cloud stretches along the valley it has discovered, and grows its step '
                         'size while progress is steady...'),
            (19.0, total, '...then shrinks as it closes in on the optimum. That adaptation is the '
                          '"CMA" in CMA-ES.')]

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Inside CMA-ES', t, 'Covariance Matrix Adaptation Evolution Strategy')
        img.paste(land, (ox, oy))
        # Draw the search state on a copy and paste back only the panel area (clipping)
        full_d = d
        layer = img.copy()
        d = ImageDraw.Draw(layer)
        gi = int(np.searchsorted(starts, t, side='right') - 1)
        gi = max(0, min(gi, len(gens) - 1))
        g = gens[gi]
        dur = times[gi] if gi < len(times) else fast_dur
        u = (t - starts[gi]) / dur if t >= starts[0] else 0
        u = max(0.0, min(1.0, u)) if t < starts[-1] else 1.0
        # mean trail
        trail = [to_px(gens[0]['mean0'])] + [to_px(gg['mean1']) for gg in gens[:gi]]
        if len(trail) > 1:
            d.line(trail, fill=(255, 255, 255), width=2)
        # ellipse interpolated between before and after update
        m = ease((u - 0.6) / 0.4)
        mean = g['mean0'] * (1 - m) + g['mean1'] * m
        cov = g['cov0'] * (1 - m) + g['cov1'] * m
        d.line(ellipse_pts(mean, cov) + [ellipse_pts(mean, cov)[0]], fill=BLUE, width=3)
        # samples
        if u > 0.05:
            order = np.argsort(g['fit'])
            best = set(order[:len(order) // 2].tolist())
            sel = ease((u - 0.3) / 0.25)
            for i, s in enumerate(g['sols']):
                px, py = to_px(s)
                col = mix(FG, AMBER, sel) if i in best else mix(FG, MUTED, sel)
                r = 6 if i in best else 5
                d.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=BG)
        mx, my = to_px(mean)
        d.ellipse([mx - 7, my - 7, mx + 7, my + 7], fill=BLUE, outline=FG, width=2)
        ex, ey = to_px(opt)
        d.line([ex - 9, ey - 9, ex + 9, ey + 9], fill=GREEN, width=3)
        d.line([ex - 9, ey + 9, ex + 9, ey - 9], fill=GREEN, width=3)
        img.paste(layer.crop((ox, oy, ox + pw, oy + ph)), (ox, oy))
        d = full_d
        d.rectangle([ox, oy, ox + pw, oy + ph], outline=PANEL, width=2)

        # side panel
        x0 = 810
        text(d, (x0, 140), f'generation {gi + 1}', 30, FG, bold=True)
        text(d, (x0, 186), f'step size σ = {g["sigma"]:.3f}', 22, MUTED, mono=True)
        text(d, (x0, 216), f'best cost   = {g["fit"].min():.3f}', 22, MUTED, mono=True)
        phases = [('sample λ = 12 candidates', u < 0.3),
                  ('select the best μ = 6', 0.3 <= u < 0.6),
                  ('move mean toward winners', 0.6 <= u),
                  ('adapt covariance & σ', 0.6 <= u)]
        for i, (lab, on) in enumerate(phases):
            y = 280 + i * 44
            d.ellipse([x0, y + 4, x0 + 18, y + 22], fill=AMBER if on else PANEL)
            text(d, (x0 + 32, y), lab, 22, FG if on else MUTED, bold=on)
        legend = [(FG, 'candidate'), (AMBER, 'selected (best half)'), (BLUE, 'mean + 1.5σ ellipse'),
                  (GREEN, 'true optimum')]
        for i, (col, lab) in enumerate(legend):
            y = 480 + i * 30
            d.ellipse([x0, y + 4, x0 + 14, y + 18], fill=col)
            text(d, (x0 + 26, y), lab, 18, MUTED)
        caption(img, caps, t)
        return img
    return float(total), render


def scene_fitness():
    lines = [('fitness =', FG, 0.4, None),
             ('    average episode reward', BLUE, 1.2,
              'environment reward: + moving toward the pad, + soft touchdown, − fuel, − crashing'),
             ('  + 100 × landing rate', GREEN, 4.5, 'a large bonus for actually landing'),
             ('  + 30 × speed × landing rate', AMBER, 7.5,
              'a bonus for landing quickly; it only counts when the lander lands')]
    caps = [(0.3, 4.5, 'Each controller is flown for 10 episodes. Its fitness combines three terms.'),
            (4.5, 7.5, 'Landing successfully is worth a lot, so evolution quickly favours pilots '
                       'that touch down.'),
            (7.5, 12.5, 'This is the default "balanced" mode. Other modes favour speed or fuel.')]

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Fitness: what "good" means', t)
        for i, (s, col, start, note) in enumerate(lines):
            a = fade(t, start, 0.5)
            y = 170 + i * 95
            text(d, (110, y), s, 40, col, a, mono=True, bold=True)
            if note:
                text(d, (140, y + 52), note, 20, MUTED, a)
        text(d, (110, 560), 'speed = 1 − average episode length / 1000 steps', 22, MUTED,
             fade(t, 8.5), mono=True)
        caption(img, caps, t)
        return img
    return 12.5, render


def scene_training(history):
    gen = np.array(history['generation'])
    best = np.array(history['best_fitness'])
    mean = np.array(history['mean_fitness'])
    land = np.array(history['best_landing_rate']) * 100
    best_run = np.maximum.accumulate(best)
    caps = [(0.3, 5.0, f'Training: {len(gen)} generations × 50 controllers × 10 episodes each.'),
            (5.0, 10.0, 'Amber: best controller per generation. Blue: population average. '
                        'Green: landing rate of the best.'),
            (10.0, 14.0, 'Within a few dozen generations the population goes from crashing to '
                         'landing reliably.')]
    px0, py0, pw, ph = 130, 150, 1000, 390
    ylo = min(-300.0, float(mean.min()))
    yhi = max(350.0, float(best.max()) + 20)

    def X(g):
        return px0 + (g - gen[0]) / max(1, gen[-1] - gen[0]) * pw

    def Y(v):
        return py0 + ph - (v - ylo) / (yhi - ylo) * ph

    def YL(v):
        return py0 + ph - v / 100 * ph

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Training progress', t)
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=PANEL, width=2)
        for v in np.arange(np.ceil(ylo / 100) * 100, yhi, 100):
            y = Y(v)
            d.line([px0, y, px0 + pw, y], fill=PANEL)
            text(d, (px0 - 10, y), f'{v:.0f}', 15, MUTED, anchor='rm')
        for p in (0, 50, 100):
            text(d, (px0 + pw + 10, YL(p)), f'{p}%', 15, GREEN, anchor='lm')
        for g in range(0, int(gen[-1]) + 1, 50):
            text(d, (X(g), py0 + ph + 12), str(g), 15, MUTED, anchor='ma')
        text(d, (px0 + pw / 2, py0 + ph + 36), 'generation', 16, MUTED, anchor='ma')
        text(d, (px0 - 60, py0 - 30), 'fitness', 16, MUTED)
        text(d, (px0 + pw + 40, py0 - 30), 'landing rate', 16, GREEN, anchor='ra')
        k = max(2, int(len(gen) * ease((t - 0.5) / 9.0)))
        for series, yf, col, wd in ((mean, Y, BLUE, 2), (land, YL, GREEN, 2), (best, Y, AMBER, 3)):
            pts = [(X(g), yf(v)) for g, v in zip(gen[:k], series[:k])]
            d.line(pts, fill=col, width=wd)
        text(d, (px0 + pw / 2, py0 - 32), f'best fitness so far: {best_run[k - 1]:.0f}   '
             f'(generation {int(gen[k - 1])})', 20, AMBER, bold=True, anchor='ma')
        caption(img, caps, t)
        return img
    return 14.0, render


def scene_evolution(checkpoints, best_path, seed):
    picks = []
    rng_ctrl = LinearController()
    rng_ctrl.weights = np.random.default_rng(0).standard_normal((8, 4)) * 0.1
    picks.append(('random start', rng_ctrl))
    wanted = [10, 30, 60, 100]
    for w in wanted:
        if checkpoints:
            g = min(checkpoints, key=lambda c: abs(c - w))
            picks.append((f'generation {g}', load_controller(checkpoints[g])))
    picks.append(('final best', load_controller(best_path)))
    eps = [record(c, seed=seed, max_steps=1000) for _, c in picks]
    caps = [(0.3, 5.5, 'The same starting conditions, flown by the best controller at different '
                       'points in training.'),
            (5.5, 11.0, 'Early pilots fall or spin away. Later ones learn to hover, then to steer '
                        'toward the pad.'),
            (11.0, 18.0, 'The final controller descends, corrects its drift and touches down '
                         'between the flags.')]
    tw, th, gx, gy = 318, 212, 28, 50
    x0, y0 = (W - (3 * tw + 2 * gx)) // 2, 135

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Evolution in action', t)
        step = int(max(0, t - 0.5) * FPS * 1.6)
        for i, ((label, _), ep) in enumerate(zip(picks, eps)):
            r, c = divmod(i, 3)
            x, y = x0 + c * (tw + gx), y0 + r * (th + gy)
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            border = (GREEN if ep['landed'] else RED) if done else PANEL
            paste_frame(img, ep['frames'][k], (x, y, tw, th), border=border)
            text(d, (x, y - 26), label, 19, AMBER, bold=True)
            if done:
                res = 'landed' if ep['landed'] else ('crashed' if ep['steps'] < 1000 else 'timed out')
                text(d, (x + tw, y - 24), f'{res}  {ep["total"]:+.0f}', 17,
                     GREEN if ep['landed'] else RED, anchor='ra')
        caption(img, caps, t)
        return img
    return 18.0, render


def describe_weight(i, j, w):
    pos, neg = STATE_MEANING[i]
    act = ACTION_NAMES[j]
    if w > 0:
        return f'when {pos}, it votes for "{act}"'
    if neg:
        return f'when {neg}, it votes for "{act}"'
    return f'when {pos}, it votes against "{act}"'


def scene_weights(best_ctrl, states):
    Wm, b = best_ctrl.weights, best_ctrl.bias
    vmax = np.abs(Wm).max()
    # Influence = |weight| x how much that input varies while airborne (legs excluded)
    states = np.asarray(states)
    airborne = states[(states[:, 6] == 0) & (states[:, 7] == 0)]
    influence = np.abs(Wm) * np.std(airborne, axis=0)[:, None]
    influence[6:] = 0
    flat = np.argsort(influence.ravel())[::-1][:3]
    tops = [divmod(int(k), 4) for k in flat]
    starts = [4.5, 8.5, 12.5]
    caps = [(0.3, 4.5, 'Because the pilot is linear, we can read what evolution discovered directly '
                       'from the weights. Here are its three most influential rules in flight.')]
    for (i, j), s in zip(tops, starts):
        caps.append((s, s + 4.0, f'{STATE_NAMES[i]} → {ACTION_NAMES[j]} ({Wm[i, j]:+.1f}): '
                                 + describe_weight(i, j, Wm[i, j]) + '.'))
    cw, chh = 120, 46
    x0, y0 = 360, 160

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'What it learned', t)
        hi = None
        for idx, s in enumerate(starts):
            if s <= t < s + 4.0:
                hi = tops[idx]
        for j in range(4):
            text(d, (x0 + j * cw + cw // 2, y0 - 22), ACTION_NAMES[j], 18, MUTED, anchor='mm')
        for i in range(8):
            y = y0 + i * chh
            text(d, (x0 - 16, y + chh // 2), STATE_NAMES[i], 19, FG, anchor='rm')
            for j in range(4):
                x = x0 + j * cw
                dimmed = hi is not None and (i, j) != hi
                col = weight_color(Wm[i, j], vmax)
                if dimmed:
                    col = mix(col, BG, 0.6)
                d.rectangle([x, y, x + cw - 4, y + chh - 4], fill=col)
                text(d, (x + cw // 2 - 2, y + chh // 2 - 2), f'{Wm[i, j]:+.2f}', 17,
                     FG, 0.4 if dimmed else 1.0, mono=True, anchor='mm', bg=col)
                if (i, j) == hi:
                    d.rectangle([x - 3, y - 3, x + cw, y + chh], outline=FG, width=3)
        y = y0 + 8 * chh + 10
        text(d, (x0 - 16, y + chh // 2), 'bias', 19, MUTED, anchor='rm')
        for j in range(4):
            x = x0 + j * cw
            col = weight_color(b[j], vmax)
            d.rectangle([x, y, x + cw - 4, y + chh - 4], fill=col)
            text(d, (x + cw // 2 - 2, y + chh // 2 - 2), f'{b[j]:+.2f}', 17, FG, mono=True,
                 anchor='mm', bg=col)
        caption(img, caps, t)
        return img
    return 16.5, render


def scene_outro(stats):
    caps = [(0.3, 6.0, 'Evaluated on 100 new episodes the controller was never trained on.'),
            (6.0, 11.0, 'Try it yourself: clone the repo, run train_ga.py, and watch your own '
                        'pilot evolve.')]
    items = [(f'{stats["landing_rate"]:.0f}%', 'landing rate', GREEN),
             (f'{stats["avg_reward"]:.0f}', 'average reward', AMBER),
             (f'{stats["avg_steps"]:.0f}', 'steps per episode', BLUE),
             ('36', 'parameters', FG)]

    def render(t):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Results', t)
        for i, (v, lab, col) in enumerate(items):
            x = 190 + i * 300
            a = fade(t, 0.5 + 0.3 * i)
            text(d, (x, 250), v, 72, col, a, bold=True, anchor='mm')
            text(d, (x, 310), lab, 20, MUTED, a, anchor='mm')
        a = fade(t, 6.0)
        text(d, (W // 2, 430), 'github.com/JCA374/Moonlander-Genetic', 34, FG, a, bold=True, anchor='mm')
        text(d, (W // 2, 485), 'pip install -r requirements.txt  &&  python train_ga.py --parallel 4',
             20, MUTED, a, mono=True, anchor='mm')
        caption(img, caps, t)
        return img
    return 11.5, render


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def evaluate(controller, episodes=100, seed0=10_000):
    eps = [record(controller, seed=seed0 + i, frames=False) for i in range(episodes)]
    return {'landing_rate': 100 * np.mean([e['landed'] for e in eps]),
            'avg_reward': float(np.mean([e['total'] for e in eps])),
            'avg_steps': float(np.mean([e['steps'] for e in eps]))}


def influence_states(controller, episodes=20):
    """States visited by the controller, used to rank weight influence."""
    states = []
    for i in range(episodes):
        states += record(controller, seed=20_000 + i, frames=False)['states']
    return states


def pick_showcase_seed(controller, candidates=range(1, 60)):
    """A seed where the best controller lands cleanly, for the gameplay scenes."""
    for s in candidates:
        ep = record(controller, seed=s, frames=False)
        if ep['landed'] and 180 < ep['steps'] < 450:
            return s
    return 1


def save_stills(scenes, stills, out_dir):
    """Write caption-free still images for the web page."""
    global SHOW_CAPTIONS
    renders = dict(scenes)
    SHOW_CAPTIONS = False
    for sname, (scene, t) in stills.items():
        renders[scene][1](t).save(os.path.join(out_dir, f'{sname}.jpg'), quality=88)
    SHOW_CAPTIONS = True


def main():
    ap = argparse.ArgumentParser(description='Render the captioned explainer video.')
    ap.add_argument('--models', default=os.path.join(REPO_ROOT, 'ga_models'))
    ap.add_argument('--output', default=os.path.join(REPO_ROOT, 'docs', 'explainer.mp4'))
    ap.add_argument('--stills', default=os.path.join(REPO_ROOT, 'docs', 'img'),
                    help='Directory for poster + still images used by the web page')
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--stills-only', action='store_true', help='Only write the still images')
    args = ap.parse_args()

    best_path = os.path.join(args.models, 'ga_best.npy')
    with open(os.path.join(args.models, 'training_history.json')) as fh:
        history = json.load(fh)
    best = load_controller(best_path)
    checkpoints = find_checkpoints(args.models)

    print('Evaluating best controller on 100 episodes...')
    stats = evaluate(best)
    print(f'  {stats}')
    seed = pick_showcase_seed(best)
    best_ep = record(best, seed=seed)
    print(f'Showcase seed {seed}: {best_ep["steps"]} steps, reward {best_ep["total"]:.1f}')

    print('Preparing scenes...')
    scenes = [
        ('title', scene_title(best_ep)),
        ('task', scene_task(best_ep)),
        ('controller', scene_controller(best, best_ep)),
        ('population', scene_population()),
        ('loop', scene_loop()),
        ('cmaes', scene_cmaes()),
        ('fitness', scene_fitness()),
        ('training', scene_training(history)),
        ('evolution', scene_evolution(checkpoints, best_path, seed)),
        ('weights', scene_weights(best, influence_states(best))),
        ('outro', scene_outro(stats)),
    ]
    # (scene name, time) of stills for the web page
    stills = {'poster': ('title', 3.0), 'controller': ('controller', 7.0),
              'population': ('population', 15.5), 'cmaes': ('cmaes', 9.0),
              'training': ('training', 13.5), 'evolution': ('evolution', 17.5),
              'weights': ('weights', 3.5)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.stills, exist_ok=True)
    save_stills(scenes, stills, args.stills)
    with open(os.path.join(args.stills, 'stats.json'), 'w') as fh:
        json.dump({**stats, 'generations': len(history['generation']),
                   'best_fitness': float(max(history['best_fitness']))}, fh, indent=2)
    if args.stills_only:
        return
    total = sum(dur for _, (dur, _) in scenes)
    print(f'Rendering {total:.1f}s of video → {args.output}')
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow',
         '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.output],
        stdin=subprocess.PIPE)
    fade_frames = 8
    for name, (dur, render) in scenes:
        n = int(round(dur * FPS))
        print(f'  {name:<11} {dur:5.1f}s')
        for i in range(n):
            t = i / FPS
            img = render(t)
            arr = np.asarray(img, dtype=np.float32)
            edge = min(i, n - 1 - i)
            if edge < fade_frames:
                arr *= (edge + 1) / (fade_frames + 1)
            ff.stdin.write(arr.astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print('Done.')


if __name__ == '__main__':
    main()
