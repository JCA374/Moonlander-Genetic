"""
Scenes for the explainer and lesson videos.

Every scene factory returns a `timeline.Scene`: a render function `render(t, beats)` plus the
default length of each beat. A beat is one narrated section (see timeline.py), so the visuals are
timed by `beats.at(k)` (when beat k starts) instead of fixed seconds.

Frames are drawn with Pillow; gameplay comes from gymnasium's rgb_array renderer.
"""

import glob
import hashlib
import json
import os
import pickle
import re
import sys

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
from timeline import Scene  # noqa: E402

W, H, FPS = 1280, 720, 30

BG = (11, 16, 32)
PANEL = (22, 30, 54)
FG = (232, 236, 245)
MUTED = (140, 152, 180)
AMBER = (255, 184, 64)
BLUE = (96, 168, 255)
GREEN = (84, 212, 140)
RED = (240, 96, 96)
ACTION_COLORS = [MUTED, (190, 120, 255), AMBER, (64, 210, 220)]

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
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
    ('the left leg is down', None),
    ('the right leg is down', None),
]
# LunarLander render geometry (600x400 frame, 30 px per metre)
VIEW_W, VIEW_H, SCALE = 600, 400, 30.0


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


def frame_at(frames, t, speed=1.0, loop_pause=1.5):
    """Gameplay frame for time t, looping with a pause on the last frame."""
    n = len(frames)
    k = int(t * FPS * speed) % (n + int(loop_pause * FPS))
    return frames[min(k, n - 1)]


def outcome(ep, max_steps=1000):
    if ep['landed']:
        return 'landed', GREEN
    return ('crashed', RED) if ep['steps'] < max_steps else ('timed out', MUTED)


def heatmap(d, Wm, b, x0, y0, cw, ch, vmax, labels=True, values=True, dim=None, outline=(),
            names=STATE_NAMES, a=1.0, size=17):
    """The 8x4 weight matrix plus bias row. `dim(i, j)` -> 0..1 dimming, `outline` cells to box."""
    if labels:
        for j in range(4):
            text(d, (x0 + j * cw + cw // 2, y0 - 22), ACTION_NAMES[j], 18, MUTED, a, anchor='mm')
    rows = [(i, Wm[i]) for i in range(8)] + [(8, b)]
    for i, row in rows:
        y = y0 + i * ch + (10 if i == 8 else 0)
        if labels:
            lab = 'bias' if i == 8 else names[i]
            text(d, (x0 - 16, y + ch // 2), lab, 19, MUTED if i == 8 else FG, a, anchor='rm')
        for j in range(4):
            x = x0 + j * cw
            col = weight_color(row[j], vmax)
            dm = dim(i, j) if dim else 0.0
            col = mix(mix(BG, col, a), BG, 0.65 * dm)
            d.rectangle([x, y, x + cw - 4, y + ch - 4], fill=col)
            if values:
                text(d, (x + cw // 2 - 2, y + ch // 2 - 2), f'{row[j]:+.2f}', size, FG,
                     a * (1 - 0.6 * dm), mono=True, anchor='mm', bg=col)
            if (i, j) in outline:
                d.rectangle([x - 3, y - 3, x + cw, y + ch], outline=FG, width=3)


def title_card(name, kicker, title, sub, dur=5.0):
    """A lesson's opening card."""
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 250), kicker, 26, AMBER, fade(t, 0.1), bold=True, anchor='mm')
        text(d, (W // 2, 330), title, 60, FG, fade(t, 0.3, 0.6), bold=True, anchor='mm')
        d.rectangle([W // 2 - int(80 * fade(t, 0.5)), 380, W // 2 + int(80 * fade(t, 0.5)), 384], fill=AMBER)
        text(d, (W // 2, 430), sub, 24, MUTED, fade(t, 0.7, 0.6), anchor='mm')
        return img
    return Scene(name, render, [dur])


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

def load_controller(path):
    c = LinearController()
    c.set_parameters(np.load(path))
    return c


def controller_from(params):
    c = LinearController()
    c.set_parameters(np.asarray(params, dtype=float))
    return c


def record(controller, seed, max_steps=1000, frames=True, path=False):
    env = gym.make('LunarLander-v2', render_mode='rgb_array' if frames else None)
    state, _ = env.reset(seed=seed)
    out = {'frames': [], 'states': [], 'actions': [], 'path': [], 'total': 0.0}
    terminated = False
    reward = 0.0
    for _ in range(max_steps):
        if frames:
            out['frames'].append(env.render())
        if path:
            p = env.unwrapped.lander.position
            out['path'].append((p.x * SCALE, VIEW_H - p.y * SCALE))
        action = controller.get_action(state)
        out['states'].append(state.copy())
        out['actions'].append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        out['total'] += reward
        if terminated or truncated:
            break
    if frames:
        out['frames'].append(env.render())
    if path:
        p = env.unwrapped.lander.position
        out['path'].append((p.x * SCALE, VIEW_H - p.y * SCALE))
    env.close()
    out['landed'] = bool(terminated and reward > 0)
    out['steps'] = len(out['actions'])
    return out


def terrain_frame(seed):
    """The moon surface for `seed`, with the lander moved out of the picture."""
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env.reset(seed=seed)
    u = env.unwrapped
    for body in [u.lander] + u.legs:
        body.position = (body.position.x + 1000, body.position.y)
    frame = env.render()
    env.close()
    return frame


def evaluate(controller, episodes=100, seed0=10_000):
    eps = [record(controller, seed=seed0 + i, frames=False) for i in range(episodes)]
    return {'landing_rate': 100 * np.mean([e['landed'] for e in eps]),
            'avg_reward': float(np.mean([e['total'] for e in eps])),
            'avg_steps': float(np.mean([e['steps'] for e in eps]))}


# ----------------------------------------------------------------------------
# Shared data (cached on disk: the expensive bits only change when the model does)
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, models_dir, cache_dir):
        self.models_dir = models_dir
        self.best_path = os.path.join(models_dir, 'ga_best.npy')
        self.best = load_controller(self.best_path)
        with open(os.path.join(models_dir, 'training_history.json')) as fh:
            self.history = json.load(fh)
        self.checkpoints = {}
        for p in glob.glob(os.path.join(models_dir, 'ga_checkpoint_*.npy')):
            self.checkpoints[int(re.search(r'(\d+)\.npy$', p).group(1))] = p
        self.checkpoints = dict(sorted(self.checkpoints.items()))
        pop_path = os.path.join(models_dir, 'population_log.npz')
        self.pop = dict(np.load(pop_path)) if os.path.exists(pop_path) else None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.key = hashlib.sha1(np.load(self.best_path).tobytes()).hexdigest()[:12]

        self.stats = self.cached('stats', lambda: evaluate(self.best))
        self.seed = self.cached('seed', self.pick_showcase_seed)
        self.best_ep = record(self.best, seed=self.seed)

    def cached(self, name, fn):
        path = os.path.join(self.cache_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                key, value = pickle.load(fh)
            if key == self.key:
                return value
        value = fn()
        with open(path, 'wb') as fh:
            pickle.dump((self.key, value), fh)
        return value

    def pick_showcase_seed(self, candidates=range(1, 200)):
        """A seed where the best controller lands in the middle of the pad and then stays still
        (fewest engine frames once both legs are down), for the gameplay scenes."""
        best = None
        for s in candidates:
            ep = record(self.best, seed=s, frames=False)
            if not (ep['landed'] and 180 < ep['steps'] < 450):
                continue
            st, a = np.asarray(ep['states']), np.asarray(ep['actions'])
            settle = int(np.sum((st[:, 6] > 0.5) & (st[:, 7] > 0.5) & (a != 0)))
            key = (settle, round(abs(float(st[-1, 0])), 2))
            if best is None or key < best[0]:
                best = (key, s)
        return best[1] if best else 1

    def airborne_states(self, episodes=20):
        def states():
            out = []
            for i in range(episodes):
                out += record(self.best, seed=20_000 + i, frames=False)['states']
            s = np.asarray(out)
            return s[(s[:, 6] == 0) & (s[:, 7] == 0)]
        return self.cached('airborne', states)

    def influence(self):
        """|weight| x how much its input varies in flight (legs excluded: they're 0 in flight)."""
        inf = np.abs(self.best.weights) * np.std(self.airborne_states(), axis=0)[:, None]
        inf[6:] = 0
        return inf

    def ablation(self, episodes=30):
        """Landing rate with each of the 36 parameters set to zero, one at a time."""
        def run():
            base = self.best.get_parameters()
            seeds = [30_000 + i for i in range(episodes)]
            full = np.mean([record(self.best, s, frames=False)['landed'] for s in seeds])
            rates = np.zeros(36)
            for k in range(36):
                p = base.copy()
                p[k] = 0.0
                c = controller_from(p)
                rates[k] = np.mean([record(c, s, frames=False)['landed'] for s in seeds])
            return {'full': float(full), 'rates': rates}
        return self.cached('ablation', run)

    def ghosts(self, gens, seed):
        """Paths of every candidate of each generation in `gens`, all flying `seed`."""
        def run():
            out = {}
            for g in gens:
                out[g] = [record(controller_from(p), seed, frames=False, path=True)
                          for p in self.pop['candidates'][g]]
                for ep in out[g]:
                    del ep['states'], ep['actions']
            return out
        return self.cached(f'ghosts_{seed}_{"-".join(map(str, gens))}', run)


# ----------------------------------------------------------------------------
# Part 1: the problem and the controller
# ----------------------------------------------------------------------------

def scene_title(ctx, name='title'):
    frames = ctx.best_ep['frames']

    def render(t, b):
        img = blank()
        f = frame_at(frames, t)
        tile = Image.fromarray(f).resize((1280, 853), Image.BILINEAR).crop((0, 133, 1280, 853))
        tile = Image.blend(tile, Image.new('RGB', tile.size, BG), 0.62)
        img.paste(tile, (0, 0))
        d = ImageDraw.Draw(img)
        a = fade(t, 0.2, 0.8)
        text(d, (W // 2, 270), 'Moonlander-Genetic', 72, FG, a, bold=True, anchor='mm')
        text(d, (W // 2, 350), 'Evolving a 36-parameter lunar lander pilot with CMA-ES', 30, AMBER,
             fade(t, 0.7, 0.8), anchor='mm')
        return img
    return Scene(name, render, [7.0])


def scene_task(ctx, name='task'):
    """Beats: the game; the 8 inputs; the 4 actions."""
    frames = ctx.best_ep['frames']

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The problem', t, 'Gymnasium LunarLander-v2')
        paste_frame(img, frame_at(frames, t), (60, 140, 660, 440))
        x0 = 770
        t1, t2 = b.at(1), b.at(2)
        text(d, (x0, 140), '8 inputs (state)', 26, BLUE, fade(t, t1), bold=True)
        for i, nm in enumerate(STATE_NAMES):
            text(d, (x0 + 10, 182 + 30 * i), f'{i + 1}. {nm}', 21, FG, fade(t, t1 + 0.4 + 0.35 * i))
        text(d, (x0, 450), '4 outputs (actions)', 26, AMBER, fade(t, t2), bold=True)
        for i, nm in enumerate(ACTION_NAMES):
            text(d, (x0 + 10, 492 + 30 * i), f'{i}. {nm}', 21, FG, fade(t, t2 + 0.6 + 0.6 * i))
        return img
    return Scene(name, render, [5.5, 6.0, 5.0])


def scene_sites(ctx, name='sites', seeds=None):
    """Beats: six random landing sites flown by the best pilot; the pad is always in the middle."""
    def pick():
        out = []
        for sd in range(500, 700):
            if record(ctx.best, sd, frames=False)['landed']:
                out.append(sd)
            if len(out) == 6:
                return out
        return out
    seeds = seeds or ctx.cached('site_seeds', pick)
    tw, th, gx, gy = 360, 240, 26, 40
    x0, y0 = (W - (3 * tw + 2 * gx)) // 2, 130
    eps = []
    for sd in seeds:
        ep = record(ctx.best, sd)
        ep['frames'] = [np.asarray(Image.fromarray(f).resize((tw, th), Image.BILINEAR)) for f in ep['frames']]
        eps.append(ep)
    # pad: between chunk 4 and 6 of 11 -> x 8..12 m of 20; at a quarter of the height
    px1, px2, py = 8 / 20 * tw, 12 / 20 * tw, th - th / 4

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'A new landing site every time', t, 'random mountains, random push')
        step = int(t * FPS * 1.3)
        hl = fade(t, b.at(1), 0.6)
        for i, ep in enumerate(eps):
            r, c = divmod(i, 3)
            x, y = x0 + c * (tw + gx), y0 + r * (th + gy)
            k = min(step, len(ep['frames']) - 1)
            paste_frame(img, ep['frames'][k], (x, y, tw, th), border=PANEL)
            if hl > 0:
                col = mix(BG, AMBER, hl)
                d.line([x + px1, y + py + 6, x + px2, y + py + 6], fill=col, width=4)
                d.line([x + tw / 2, y + 4, x + tw / 2, y + py - 4], fill=mix(BG, AMBER, 0.5 * hl), width=1)
        if hl > 0:
            text(d, (W // 2, y0 + 2 * th + gy + 22), 'the pad is always in the middle, at the same height: '
                 'x = 0, y = 0 in the lander\'s inputs', 21, AMBER, hl, anchor='ma')
        return img
    longest = max(len(e['frames']) for e in eps) / (FPS * 1.3)
    return Scene(name, render, [7.0, 7.0], min_beats=[0, max(0.0, longest - 6.0)])


def scene_reward(ctx, name='reward'):
    """Beats: shaping terms; one-off events; fuel; 'solved'."""
    items = [(0, '+', 'moving closer to the pad, slowing down, staying level', GREEN),
             (0, '−', 'drifting away, speeding up, tilting', RED),
             (1, '+10', 'each leg that touches the ground', GREEN),
             (1, '+100', 'coming to rest on the ground', GREEN),
             (1, '−100', 'crashing or flying off screen', RED),
             (2, '−0.3', 'every frame the main engine fires', AMBER),
             (2, '−0.03', 'every frame a side engine fires', AMBER)]
    frames = ctx.best_ep['frames']

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The reward', t, 'what the game itself scores')
        paste_frame(img, frame_at(frames, t), (60, 150, 420, 280))
        for k, (beat, v, s, col) in enumerate(items):
            start = b.at(beat) + 0.5 + 0.9 * sum(1 for it in items[:k] if it[0] == beat)
            a = fade(t, start)
            y = 150 + k * 58
            text(d, (540, y), v, 30, col, a, bold=True, mono=True)
            text(d, (680, y + 6), s, 21, FG, a)
        a = fade(t, b.at(3))
        d.rounded_rectangle([600, 580, 1220, 640], 12, fill=mix(BG, PANEL, a))
        text(d, (910, 610), 'an episode scoring 200 or more counts as solved', 22, AMBER, a,
             anchor='mm', bg=PANEL)
        return img
    return Scene(name, render, [6.0, 6.0, 5.0, 4.0])


def scene_controller(ctx, name='controller'):
    """Beats: the matrix; scores and argmax; 36 numbers."""
    Wm, bias = ctx.best.weights, ctx.best.bias
    vmax = np.abs(Wm).max()
    frames, states, actions = ctx.best_ep['frames'], ctx.best_ep['states'], ctx.best_ep['actions']
    n = len(states)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The controller: 36 numbers', t)
        k = min(int(t * FPS) % (n + 45), n - 1)
        s, act = states[k], actions[k]
        paste_frame(img, frames[k], (40, 150, 510, 340))
        text(d, (295, 510), f'action: {ACTION_NAMES[act]}', 26, ACTION_COLORS[act], bold=True, anchor='ma')

        x_lab, x_bar, x_hm, row0, rh = 580, 700, 850, 175, 40
        text(d, (x_lab, 122), 'state', 18, BLUE, bold=True)
        text(d, (x_hm + 4 * 47 // 2, 122), 'weights W', 18, MUTED, bold=True, anchor='ma')
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
            d.rectangle([cx0, yb, cx0 + 44, yb + 26], fill=weight_color(bias[j], vmax))

        a = fade(t, b.at(1))
        scores = s @ Wm + bias
        xs = 1065
        text(d, (xs, 122), 'scores', 18, AMBER, a, bold=True)
        smax = max(1e-6, np.abs(scores).max())
        for j in range(4):
            y = row0 + j * 70
            win = j == int(np.argmax(scores))
            text(d, (xs, y), ACTION_SHORT[j], 18, AMBER if win else FG, a, bold=win)
            d.rectangle([xs, y + 26, xs + 130, y + 50], fill=mix(BG, PANEL, a))
            wbar = int(abs(scores[j]) / smax * 130 * a)
            d.rectangle([xs, y + 26, xs + wbar, y + 50], fill=AMBER if win else MUTED)
            text(d, (xs + 136, y + 28), f'{scores[j]:+.1f}', 15, FG, a)
        text(d, (xs, row0 + 4 * 70 + 10), 'argmax → action', 17, MUTED, a)

        a = fade(t, b.at(2), 0.6)
        if a > 0:
            d.rounded_rectangle([580, 580, 1240, 640], 12, fill=mix(BG, PANEL, a))
            text(d, (910, 610), '8 × 4 weights  +  4 biases  =  36 parameters', 26, AMBER, a,
                 bold=True, anchor='mm', bg=PANEL)
        return img
    return Scene(name, render, [6.0, 5.0, 5.0])


def scene_anatomy(ctx, name='w_anatomy'):
    """Beats: the grid; one row (an input); one column (an action); one weight read out; bias."""
    Wm, bias = ctx.best.weights, ctx.best.bias
    vmax = np.abs(Wm).max()
    inf = ctx.influence()
    i0, j0 = np.unravel_index(int(np.argmax(inf)), inf.shape)
    x0, y0, cw, ch = 400, 150, 130, 46

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Reading the weight matrix', t)
        k = b.index(t)
        dim, outline, note = None, (), ''
        if k == 1:
            dim = lambda i, j: 0.0 if i == i0 else 1.0  # noqa: E731
            note = f'one row = one input ({STATE_NAMES[i0]}) voting on all four actions'
        elif k == 2:
            dim = lambda i, j: 0.0 if j == j0 else 1.0  # noqa: E731
            note = f'one column = everything that pushes "{ACTION_NAMES[j0]}" up or down'
        elif k == 3:
            dim = lambda i, j: 0.0 if (i, j) == (i0, j0) else 1.0  # noqa: E731
            outline = [(i0, j0)]
            w = Wm[i0, j0]
            note = f'{w:+.2f}: {"positive" if w > 0 else "negative"} — ' + describe_weight(i0, j0, w)
        elif k == 4:
            dim = lambda i, j: 0.0 if i == 8 else 1.0  # noqa: E731
            note = 'the bias: a constant head start for each action, whatever the state'
        heatmap(d, Wm, bias, x0, y0, cw, ch, vmax, dim=dim, outline=outline)
        # legend
        lx = 960
        for v, lab in ((vmax, 'positive: pushes the score up'), (-vmax, 'negative: pushes it down')):
            ly = 170 if v > 0 else 210
            d.rectangle([lx, ly, lx + 26, ly + 24], fill=weight_color(v, vmax))
            text(d, (lx + 38, ly + 2), lab, 17, MUTED)
        text(d, (lx, 250), 'brighter = larger', 17, MUTED)
        if note:
            a = fade(t, b.at(k) + 0.2)
            d.rounded_rectangle([80, 620, 1200, 670], 12, fill=mix(BG, PANEL, a))
            text(d, (640, 645), note, 21, FG, a, anchor='mm', bg=PANEL)
        return img
    return Scene(name, render, [5.0, 5.5, 5.5, 6.0, 5.0])


def describe_weight(i, j, w):
    pos, neg = STATE_MEANING[i]
    act = ACTION_NAMES[j]
    if w > 0:
        return f'when {pos}, it votes for "{act}"'
    if neg:
        return f'when {neg}, it votes for "{act}"'
    return f'when {pos}, it votes against "{act}"'


# ----------------------------------------------------------------------------
# Part 2: the genetic algorithm
# ----------------------------------------------------------------------------

def scene_population(ctx, name='population', count=18, seed=7):
    """Beats: random pilots; they fly and get scored; the best half is kept."""
    def run():
        rng = np.random.default_rng(seed)
        eps = []
        for i in range(count):
            ep = record(controller_from(rng.standard_normal(36) * 0.5), seed=100 + i, max_steps=600)
            ep['frames'] = [np.asarray(Image.fromarray(f).resize((165, 110), Image.BILINEAR))
                            for f in ep['frames']]
            del ep['states'], ep['actions']
            eps.append(ep)
        return eps
    eps = ctx.cached(f'population_{seed}_{count}', run)
    scores = [e['total'] + (100 if e['landed'] else 0) for e in eps]
    selected = set(np.argsort(scores)[::-1][:count // 2].tolist())
    tw, th, gap = 165, 110, 16
    x0 = (W - (6 * tw + 5 * gap)) // 2
    y0 = 140

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'A population of pilots', t, 'generation 0: 36 random numbers each')
        fly = max(0.0, t - b.at(1) + 0.5) if b.at(1) < 1e9 else t
        step = int(fly * FPS * 2)
        sel_a = fade(t, b.at(2), 0.8)
        for i, ep in enumerate(eps):
            r, c = divmod(i, 6)
            x, y = x0 + c * (tw + gap), y0 + r * (th + 50)
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            dim = 0.65 * sel_a if i not in selected else 0.0
            border = mix(BG, AMBER, sel_a) if (sel_a > 0 and i in selected) else None
            paste_frame(img, ep['frames'][k], (x, y, tw, th), border=border, dim=dim)
            if done:
                lab, col = outcome(ep, 600)
                text(d, (x + 4, y + th + 8), lab, 16, col, 1 - 0.6 * (dim > 0))
                text(d, (x + tw - 4, y + th + 8), f'{scores[i]:+.0f}', 16, FG, 1 - 0.6 * (dim > 0),
                     anchor='ra')
        a = fade(t, b.at(2) + 0.8)
        text(d, (W // 2, 650), 'kept: the best half (amber)  →  they shape the next generation', 21,
             AMBER, a, anchor='mm')
        return img
    return Scene(name, render, [4.5, 6.0, 5.0], min_beats=[0, 5.5, 0])


def scene_loop(ctx, name='loop'):
    steps = [('Sample', 'draw a population of 36-number\ncontrollers from a Gaussian'),
             ('Evaluate', 'fly each one for several\nepisodes, compute fitness'),
             ('Select', 'rank them, keep\nthe best half'),
             ('Update', 'move the mean toward the winners,\nreshape the covariance')]
    cx, cy, R = W // 2, 390, 200
    pos = [(cx, cy - R), (cx + R + 110, cy), (cx, cy + R), (cx - R - 110, cy)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The evolutionary loop', t)
        active = int(t / 1.3) % 4
        d.ellipse([cx - R - 110, cy - R, cx + R + 110, cy + R], outline=PANEL, width=4)
        for i, (nm, desc) in enumerate(steps):
            px, py = pos[i]
            on = i == active
            d.rounded_rectangle([px - 150, py - 48, px + 150, py + 48], 14,
                                fill=mix(PANEL, AMBER, 0.25 if on else 0), outline=AMBER if on else MUTED,
                                width=3 if on else 1)
            text(d, (px, py - 30), f'{i + 1}. {nm}', 24, AMBER if on else FG, bold=True,
                 anchor='ma', bg=PANEL)
            for li, line in enumerate(desc.split('\n')):
                text(d, (px, py + 2 + 20 * li), line, 16, FG, anchor='ma', bg=PANEL)
        text(d, (cx, cy), f'generation {int(t / 5.2) + 1}', 22, MUTED, anchor='mm')
        return img
    return Scene(name, render, [10.0] * 1)


def _toy_cmaes():
    ang = np.deg2rad(35)
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    opt = np.array([2.0, 1.5])

    def f(p):
        u = Rm.T @ (np.asarray(p) - opt)
        return u[0] ** 2 + 25 * u[1] ** 2

    es = cma.CMAEvolutionStrategy([-3.0, -2.6], 0.6, {'popsize': 12, 'seed': 3, 'verbose': -9})
    gens = []
    for _ in range(40):
        mean0, cov0 = es.mean.copy(), es.sigma ** 2 * np.array(es.C)
        sols = es.ask()
        fit = [f(s) for s in sols]
        es.tell(sols, fit)
        gens.append({'sols': np.array(sols), 'fit': np.array(fit), 'mean0': mean0, 'cov0': cov0,
                     'mean1': es.mean.copy(), 'cov1': es.sigma ** 2 * np.array(es.C),
                     'sigma': es.sigma})
        if es.sigma < 0.01:
            break
    return f, opt, gens


def scene_cmaes(ctx, name='cmaes', slow=6):
    """Beats: the landscape; the first `slow` generations, step by step; the rest, fast."""
    f, opt, gens = _toy_cmaes()
    pw, ph = 700, 520
    xmin, xmax, ymin, ymax = -4.5, 4.0, -3.6, 3.0
    xs, ys = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 240))
    Z = np.vectorize(lambda a, c: f((a, c)))(xs, ys)
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
    ox, oy = 60, 135

    def to_px(p):
        return (ox + (p[0] - xmin) / (xmax - xmin) * pw, oy + (ymax - p[1]) / (ymax - ymin) * ph)

    def ellipse_pts(mean, cov, k=1.5):
        vals, vecs = np.linalg.eigh(cov)
        th = np.linspace(0, 2 * np.pi, 64)
        pts = (vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12)) * k) @ np.vstack([np.cos(th), np.sin(th)])).T
        return [to_px(mean + p) for p in pts]

    def gen_time(t, b):
        """(generation index, progress 0..1 within it); -1 before the search starts."""
        if t < b.at(1):
            return -1, 0.0
        if t < b.at(2):
            dur = (b.at(2) - b.at(1)) / slow
            g = int((t - b.at(1)) / dur)
            return min(g, slow - 1), (t - b.at(1)) / dur - g
        rest = len(gens) - slow
        dur = max(0.2, (b.end - b.at(2) - 1.5) / rest)
        u = (t - b.at(2)) / dur
        g = slow + int(u)
        if g >= len(gens):
            return len(gens) - 1, 1.0
        return g, u - int(u)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Inside CMA-ES', t, 'Covariance Matrix Adaptation Evolution Strategy')
        img.paste(land, (ox, oy))
        full_d = d
        layer = img.copy()
        d = ImageDraw.Draw(layer)
        gi, u = gen_time(t, b)
        g = gens[max(gi, 0)]
        trail = [to_px(gens[0]['mean0'])] + [to_px(gg['mean1']) for gg in gens[:max(gi, 0)]]
        if len(trail) > 1:
            d.line(trail, fill=(255, 255, 255), width=2)
        m = ease((u - 0.6) / 0.4) if gi >= 0 else 0.0
        mean = g['mean0'] * (1 - m) + g['mean1'] * m
        cov = g['cov0'] * (1 - m) + g['cov1'] * m
        d.line(ellipse_pts(mean, cov) + [ellipse_pts(mean, cov)[0]], fill=BLUE, width=3)
        if gi >= 0 and u > 0.05:
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

        x0 = 810
        text(d, (x0, 150), f'generation {max(gi, 0) + 1}', 30, FG, bold=True)
        text(d, (x0, 196), f'step size σ = {g["sigma"]:.3f}', 22, MUTED, mono=True)
        text(d, (x0, 226), f'best cost   = {g["fit"].min():.3f}', 22, MUTED, mono=True)
        on = [gi >= 0 and u < 0.3, gi >= 0 and 0.3 <= u < 0.6, gi >= 0 and u >= 0.6, gi >= 0 and u >= 0.6]
        for i, lab in enumerate(['sample λ = 12 candidates', 'select the best μ = 6',
                                 'move mean toward winners', 'adapt covariance & σ']):
            y = 290 + i * 44
            d.ellipse([x0, y + 4, x0 + 18, y + 22], fill=AMBER if on[i] else PANEL)
            text(d, (x0 + 32, y), lab, 22, FG if on[i] else MUTED, bold=on[i])
        for i, (col, lab) in enumerate([(FG, 'candidate'), (AMBER, 'selected (best half)'),
                                        (BLUE, 'mean + 1.5σ ellipse'), (GREEN, 'true optimum')]):
            y = 490 + i * 30
            d.ellipse([x0, y + 4, x0 + 14, y + 18], fill=col)
            text(d, (x0 + 26, y), lab, 18, MUTED)
        text(d, (ox, oy + ph + 14), 'toy problem in 2-D: dark = low cost (good), bright = high cost',
             17, MUTED)
        return img
    return Scene(name, render, [5.0, 9.6, 9.0], min_beats=[0, 9.6, 8.0])


def scene_cmaes_cov(ctx, name='cmaes_cov'):
    """Beats: what the Gaussian is; mean; sigma; covariance — each part highlighted on one ellipse."""
    rng = np.random.default_rng(4)
    cov = np.array([[1.6, 1.05], [1.05, 0.9]])
    pts = rng.multivariate_normal([0, 0], cov, 200)
    ox, oy, sc = 360, 400, 95
    pts = pts[(np.abs(pts[:, 0]) < 3.0) & (np.abs(pts[:, 1]) < 2.5)]

    def to_px(p):
        return (ox + p[0] * sc, oy - p[1] * sc)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The search distribution', t, 'a Gaussian cloud: mean, step size σ, covariance C')
        k = b.index(t)
        n = int(len(pts) * ease((t - 0.3) / 2.0))
        for p in pts[:n]:
            x, y = to_px(p)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=mix(BG, FG, 0.55))
        vals, vecs = np.linalg.eigh(cov)
        th = np.linspace(0, 2 * np.pi, 80)
        for kk, wd in ((1.0, 2), (2.0, 1)):
            e = (vecs @ np.diag(np.sqrt(vals) * kk) @ np.vstack([np.cos(th), np.sin(th)])).T
            d.line([to_px(p) for p in e] + [to_px(e[0])], fill=BLUE if k >= 3 else MUTED, width=wd + (k >= 3))
        mx, my = to_px((0, 0))
        hl = k >= 1
        d.ellipse([mx - 8, my - 8, mx + 8, my + 8], fill=AMBER if hl else BLUE, outline=FG, width=2)
        if k >= 2:
            r = 0.8 * sc
            d.ellipse([mx - r, my - r, mx + r, my + r], outline=GREEN, width=3)
            d.line([mx, my, mx + r, my], fill=GREEN, width=3)
            text(d, (mx + r / 2, my + 8), 'σ', 24, GREEN, bold=True, anchor='ma')
        if k >= 3:
            ax_ = vecs[:, 1] * np.sqrt(vals[1]) * 2.0
            d.line([to_px(-ax_), to_px(ax_)], fill=BLUE, width=3)
        notes = [('mean', 'where the search is centred: the current best guess', AMBER, 1),
                 ('σ', 'step size: how far from the mean new candidates land', GREEN, 2),
                 ('C', 'covariance: the shape. Which directions to search wider', BLUE, 3)]
        for i, (sym, s, col, kk) in enumerate(notes):
            a = fade(t, b.at(kk))
            y = 160 + i * 140
            text(d, (760, y), sym, 34, col, a, bold=True)
            for li, line in enumerate(_wrap(s, 40)):
                text(d, (760, y + 50 + 28 * li), line, 21, FG, a)
        a = fade(t, b.at(4))
        text(d, (W // 2, 660), 'in our problem the cloud has 36 dimensions: one per weight', 22,
             AMBER, a, anchor='mm')
        return img
    return Scene(name, render, [5.0, 5.0, 5.0, 6.0, 5.0])


def _wrap(s, width):
    import textwrap
    return textwrap.wrap(s, width)


def scene_fitness(ctx, name='fitness', reveal=((0, 0.1), (0, 0.45), (1, 0.0), (1, 0.35), (1, 0.7))):
    """`reveal`: (beat, fraction of that beat) at which each of the 5 lines appears."""
    lines = [('fitness =', FG, None),
             ('    average episode reward', BLUE,
              'the game\'s reward: + approaching the pad, + soft touchdown, − fuel, − crashing'),
             ('  + 100 × landing rate', GREEN, 'a large bonus for actually landing'),
             ('  + 30 × speed × landing rate', AMBER,
              'a bonus for landing quickly; it only counts when the lander lands'),
             ('  − 0.5 × engine frames after touchdown', RED,
              'once both legs are down, stay still: no more engine bursts')]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Fitness: what "good" means', t, 'each pilot flies 10 episodes')
        for i, (s, col, note) in enumerate(lines):
            bk, fr = reveal[i]
            a = fade(t, b.at(bk) + fr * b.durations[min(bk, len(b.durations) - 1)], 0.5)
            y = 150 + i * 92
            text(d, (110, y), s, 36, col, a, mono=True, bold=True)
            if note:
                text(d, (140, y + 46), note, 19, MUTED, a)
        bk, fr = reveal[3]
        text(d, (110, 640), 'speed = 1 − average episode length / 1000 steps', 20, MUTED,
             fade(t, b.at(bk) + (fr + 0.3) * b.durations[min(bk, len(b.durations) - 1)]), mono=True)
        return img
    return Scene(name, render, [6.0] * (max(r[0] for r in reveal) + 1))


# ----------------------------------------------------------------------------
# Part 3: generations (from population_log.npz)
# ----------------------------------------------------------------------------

def _gen_progress(t, b, start_beat, keyframes):
    """Generation shown at time t: `keyframes` = generation reached at the end of each beat
    from `start_beat` on (linear in between, eased)."""
    k, lt, dur = b.local(t)
    if k < start_beat:
        return keyframes[0][0]
    idx = min(k - start_beat, len(keyframes) - 1)
    g0, g1 = keyframes[idx]
    u = ease(lt / max(0.1, dur - 0.8))
    return g0 + (g1 - g0) * u


def scene_gen_fitness(ctx, name='gen_fitness', keyframes=((0, 4), (4, 35), (35, 199))):
    """Every candidate's fitness, one column of dots per generation. Beats follow `keyframes`."""
    pop = ctx.pop
    fit, land = pop['fitness'], pop['landing_rate']
    G = len(fit)
    keyframes = [(min(a, G - 1), min(c, G - 1)) for a, c in keyframes]
    px0, py0, pw, ph = 130, 140, 960, 440
    ylo, yhi = -400.0, max(400.0, float(fit.max()) + 20)

    def X(g):
        return px0 + g / max(1, G - 1) * pw

    def Y(v):
        return py0 + ph - (np.clip(v, ylo, yhi) - ylo) / (yhi - ylo) * ph

    # pre-render all dots, revealed by cropping
    dots = Image.new('RGB', (W, H), BG)
    dd = ImageDraw.Draw(dots)
    rng = np.random.default_rng(0)
    for g in range(G):
        for v, l in zip(fit[g], land[g]):
            x = X(g) + rng.uniform(-1.5, 1.5)
            y = Y(v)
            col = mix(RED, GREEN, l)
            dd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=mix(BG, col, 0.8))
    best = fit.max(axis=1)
    mean = fit.mean(axis=1)
    land_frac = (land > 0.5).mean(axis=1) * 100

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Every pilot, every generation', t, f'{G} generations × {fit.shape[1]} pilots')
        gp = _gen_progress(t, b, 0, keyframes)
        k = int(gp) + 1
        xcut = int(X(gp) + 3)
        img.paste(dots.crop((px0, py0, xcut, py0 + ph)), (px0, py0))
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=PANEL, width=2)
        for v in np.arange(np.ceil(ylo / 100) * 100, yhi, 100):
            y = Y(v)
            d.line([xcut, y, px0 + pw, y], fill=PANEL)
            text(d, (px0 - 10, y), f'{v:.0f}' if v > ylo else f'≤{v:.0f}', 15, MUTED, anchor='rm')
        for g in range(0, G, 25):
            text(d, (X(g), py0 + ph + 12), str(g), 15, MUTED, anchor='ma')
        text(d, (px0 + pw / 2, py0 + ph + 34), 'generation', 16, MUTED, anchor='ma')
        text(d, (px0 - 60, py0 - 28), 'fitness', 16, MUTED)
        if k > 1:
            show_lines = fade(t, b.at(len(b.durations) - 1))
            if show_lines > 0:
                for series, col, wd in ((mean, BLUE, 3), (best, AMBER, 3)):
                    pts = [(X(g), Y(v)) for g, v in enumerate(series[:k])]
                    d.line(pts, fill=mix(BG, col, show_lines), width=wd)
        g = min(k - 1, G - 1)
        x = 1110
        text(d, (x, 150), f'gen {g}', 30, FG, bold=True)
        text(d, (x, 200), 'pilots that land', 15, MUTED)
        text(d, (x, 220), 'most episodes', 15, MUTED)
        text(d, (x, 244), f'{land_frac[g]:.0f}%', 34, GREEN, bold=True)
        text(d, (x, 304), 'best fitness', 15, MUTED)
        text(d, (x, 326), f'{best[g]:.0f}', 30, AMBER, bold=True)
        text(d, (x, 380), 'average', 15, MUTED)
        text(d, (x, 402), f'{mean[g]:.0f}', 30, BLUE, bold=True)
        for i, (col, lab) in enumerate([(GREEN, 'dot: one pilot'), (RED, 'red: crashes'),
                                        (AMBER, 'best'), (BLUE, 'average')]):
            y = 470 + i * 26
            d.ellipse([x, y + 4, x + 12, y + 16], fill=col)
            text(d, (x + 20, y), lab, 15, MUTED)
        return img
    return Scene(name, render, [5.0, 6.0, 7.0][:len(keyframes)],
                 min_beats=[0, 5.0, 6.0][:len(keyframes)])


def scene_gen_ghosts(ctx, name='gen_ghosts', gens_by_beat=((0,), (0, 1, 5), (10, 20, 30, 40, 50), (75, 100, 199))):
    """All pilots of a generation flying the same start, drawn as trails."""
    G = len(ctx.pop['fitness'])
    gens_by_beat = [tuple(min(g, G - 1) for g in gs) for gs in gens_by_beat]
    gens = sorted({g for gs in gens_by_beat for g in gs})
    seed = ctx.seed
    paths = ctx.ghosts(gens, seed)
    bg = terrain_frame(seed)
    bx, by, bw, bh = 40, 125, 780, 520
    bgimg = Image.fromarray(bg).resize((bw, bh), Image.BILINEAR)
    bgimg = Image.blend(bgimg, Image.new('RGB', (bw, bh), BG), 0.35)
    sx, sy = bw / VIEW_W, bh / VIEW_H
    best_fit = ctx.pop['fitness'].max(axis=1)

    def slot(t, b):
        k, lt, dur = b.local(t)
        gs = gens_by_beat[min(k, len(gens_by_beat) - 1)]
        if k == 0 and len(gens_by_beat) > 1 and gens_by_beat[1][0] == gs[0]:
            return gs[0], ease(lt / max(1.0, dur)) * 0.6       # intro beat: start drawing slowly
        n = len(gs)
        per = dur / n
        i = min(int(lt / per), n - 1)
        u = (lt - i * per) / max(0.1, per * 0.75)
        if k == 1 and gs[0] == gens_by_beat[0][0] and i == 0:
            u = 0.6 + 0.4 * u                                  # continue the intro's generation
        return gs[i], min(1.0, u)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'One generation, fifty flights', t, 'every pilot starts from the same spot')
        g, u = slot(t, b)
        eps = paths[g]
        layer = bgimg.copy().convert('RGBA')
        over = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        longest = max(len(e['path']) for e in eps)
        upto = int(u * longest) + 1
        counts = {'landed': 0, 'crashed': 0, 'timed out': 0}
        for e in eps:
            p = [(x * sx, y * sy) for x, y in e['path'][:upto]]
            done = upto >= len(e['path'])
            lab, col = outcome(e)
            c = col if done else FG
            if len(p) > 1:
                od.line(p, fill=c + (120,), width=2)
            hx, hy = p[-1]
            od.ellipse([hx - 3, hy - 3, hx + 3, hy + 3], fill=c + (230,))
            if done:
                counts[lab] += 1
        layer = Image.alpha_composite(layer, over).convert('RGB')
        img.paste(layer, (bx, by))
        d.rectangle([bx, by, bx + bw, by + bh], outline=PANEL, width=2)

        x = 860
        text(d, (x, 140), 'generation', 20, MUTED)
        text(d, (x, 165), f'{g}', 64, FG, bold=True)
        for i, (lab, col) in enumerate([('landed', GREEN), ('crashed', RED), ('timed out', MUTED)]):
            y = 270 + i * 44
            text(d, (x, y), lab, 22, col)
            text(d, (x + 330, y), f'{counts[lab]}', 28, col, bold=True, anchor='ra')
        # best-fitness strip with the current generation marked
        sx0, sy0, sw, sh = x, 450, 340, 150
        d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], outline=PANEL, width=2)
        lo, hi = -400.0, max(400.0, float(best_fit.max()))
        pts = [(sx0 + i / (G - 1) * sw, sy0 + sh - (np.clip(v, lo, hi) - lo) / (hi - lo) * sh)
               for i, v in enumerate(best_fit)]
        d.line(pts, fill=mix(BG, AMBER, 0.6), width=2)
        mx = sx0 + g / (G - 1) * sw
        d.line([mx, sy0, mx, sy0 + sh], fill=FG, width=2)
        text(d, (sx0, sy0 + sh + 8), 'best fitness per generation', 15, MUTED)
        return img
    beats = [5.0, 6.0, 8.0, 6.0][:len(gens_by_beat)]
    mins = [0.0] + [2.2 * len(gs) for gs in gens_by_beat[1:]]
    return Scene(name, render, beats, min_beats=mins)


def scene_gen_cloud(ctx, name='gen_cloud'):
    """The population in parameter space (PCA of all 36-D candidates).
    Beats: gen 0; the journey; the scale trick (relative step size)."""
    C = ctx.pop['candidates']
    G, P, _ = C.shape
    flat = C.reshape(-1, 36)
    mu = flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    proj = ((flat - mu) @ Vt[:2].T).reshape(G, P, 2)
    means = (ctx.pop['mean'] - mu) @ Vt[:2].T
    sig = ctx.pop['sigma']
    dist = np.linalg.norm(ctx.pop['mean'], axis=1)
    rel = sig / np.maximum(dist, 1e-9)
    ox, oy, pw, ph = 60, 125, 760, 520
    allp = np.vstack([proj.reshape(-1, 2), means])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    c = (lo + hi) / 2
    hw = max((hi - lo)[0] / 2 * 1.08, (hi - lo)[1] / 2 * 1.08 * pw / ph)
    hh = hw * ph / pw
    bg = (16, 22, 42)

    def lp(p):
        return (pw / 2 + (p[0] - c[0]) / hw * pw / 2, ph / 2 - (p[1] - c[1]) / hh * ph / 2)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The population in parameter space', t, '36 dimensions flattened to 2 (PCA)')
        gp = _gen_progress(t, b, 0, [(0, 3), (3, G - 1), (G - 1, G - 1)][:len(b.durations)])
        g = int(gp)
        layer = Image.new('RGB', (pw, ph), bg)
        ld = ImageDraw.Draw(layer)
        for gg in range(max(0, g - 8), g):
            for p in proj[gg]:
                x, y = lp(p)
                ld.ellipse([x - 2, y - 2, x + 2, y + 2], fill=mix(bg, MUTED, 0.1 + 0.04 * (gg - g + 8)))
        trail = [lp(m) for m in means[:g + 1]]
        if len(trail) > 1:
            ld.line(trail, fill=(255, 255, 255), width=2)
        for i in range(P):
            x, y = lp(proj[g, i])
            ld.ellipse([x - 4, y - 4, x + 4, y + 4], fill=mix(RED, GREEN, ctx.pop['landing_rate'][g, i]))
        mx, my = lp(means[g])
        ld.ellipse([mx - 7, my - 7, mx + 7, my + 7], fill=BLUE, outline=FG, width=2)
        ox0, oy0 = lp(means[0])
        ld.line([ox0 - 8, oy0, ox0 + 8, oy0], fill=MUTED, width=2)
        ld.line([ox0, oy0 - 8, ox0, oy0 + 8], fill=MUTED, width=2)
        img.paste(layer, (ox, oy))
        d.rectangle([ox, oy, ox + pw, oy + ph], outline=PANEL, width=2)

        x = 860
        text(d, (x, 140), 'generation', 20, MUTED)
        text(d, (x, 165), f'{g}', 64, FG, bold=True)
        text(d, (x, 262), f'size of the weights  {dist[g]:5.1f}', 19, FG, mono=True)
        text(d, (x, 290), f'step size σ          {sig[g]:5.2f}', 19, FG, mono=True)
        for i, (col, lab) in enumerate([(GREEN, 'pilot that lands'), (RED, 'pilot that crashes'),
                                        (BLUE, 'centre of the cloud (mean)'), (FG, 'path of the mean'),
                                        (MUTED, '+ where it started')]):
            y = 340 + i * 28
            d.ellipse([x, y + 4, x + 14, y + 18], fill=col)
            text(d, (x + 26, y), lab, 17, MUTED)
        a = fade(t, b.at(2)) if len(b.durations) > 2 else 0.0
        sx0, sy0, sw, sh = x, 520, 340, 100
        if a > 0:
            text(d, (x, sy0 - 34), 'relative step  σ / size', 19, AMBER, a, bold=True)
            d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], outline=mix(BG, PANEL, a), width=2)
            lr = np.log10(np.maximum(rel, 1e-3))
            l0, l1 = lr.min(), lr.max()
            pts = [(sx0 + i / (G - 1) * sw, sy0 + 4 + (l1 - v) / (l1 - l0) * (sh - 8)) for i, v in enumerate(lr)]
            d.line(pts, fill=mix(BG, AMBER, a), width=3)
            text(d, (sx0 + sw, sy0 + sh + 6), f'{rel[0]:.2f} → {rel[-1]:.3f}', 16, MUTED, a, anchor='ra')
        return img
    return Scene(name, render, [6.0, 9.0, 7.0], min_beats=[0, 8.0, 0])


def scene_gen_weights(ctx, name='gen_weights'):
    """The CMA mean shown as the weight matrix, morphing generation by generation.
    Colours are scaled to each generation's largest weight, so the pattern shows as it forms."""
    means = ctx.pop['mean']
    G = len(means)
    inf = ctx.influence()
    top = [divmod(int(k), 4) for k in np.argsort(inf.ravel())[::-1][:4]]
    x0, y0, cw, ch = 330, 150, 120, 46

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Watching the controller form', t, 'the centre of the population, as weights')
        gp = _gen_progress(t, b, 0, [(0, 60), (60, G - 1)][:len(b.durations)])
        g = int(gp)
        m = means[g]
        vmax = np.abs(m[:32]).max()
        hl = b.index(t) >= 1
        dim = (lambda i, j: 0.0 if (i, j) in top else 0.6) if hl else None
        heatmap(d, m[:32].reshape(8, 4), m[32:], x0, y0, cw, ch, vmax, dim=dim, outline=top if hl else ())
        x = 880
        text(d, (x, 150), 'generation', 20, MUTED)
        text(d, (x, 175), f'{g}', 64, FG, bold=True)
        text(d, (x, 265), f'largest weight  {vmax:5.1f}', 19, FG, mono=True)
        if hl:
            a = fade(t, b.at(1))
            text(d, (x, 320), 'strongest rules in the end', 20, AMBER, a, bold=True)
            for n, (i, j) in enumerate(top):
                text(d, (x, 356 + 30 * n), f'{STATE_SHORT[i]} → {ACTION_SHORT[j]}', 20, FG, a)
                text(d, (x + 300, 356 + 30 * n), f'{means[-1][i * 4 + j]:+.1f}', 18, MUTED, a,
                     mono=True, anchor='ra')
        return img
    return Scene(name, render, [6.0, 8.0], min_beats=[0, 6.0])


def scene_evolution(ctx, name='evolution', wanted=(10, 30, 60, 100)):
    """Beats: the line-up; playback (best pilot at different points in training)."""
    picks = [('random start', controller_from(np.concatenate(
        [np.random.default_rng(0).standard_normal(32) * 0.1, np.zeros(4)])))]
    for w in wanted:
        if ctx.checkpoints:
            g = min(ctx.checkpoints, key=lambda c: abs(c - w))
            picks.append((f'generation {g}', load_controller(ctx.checkpoints[g])))
    picks.append(('final best', ctx.best))
    eps = [record(c, seed=ctx.seed, max_steps=1000) for _, c in picks]
    tw, th, gx, gy = 318, 212, 28, 56
    x0, y0 = (W - (3 * tw + 2 * gx)) // 2, 150

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Evolution in action', t, 'the best pilot so far, same start every time')
        play = max(0.0, t - b.at(1) + 0.3) if b.at(1) < 1e9 else max(0.0, t - 0.5)
        step = int(play * FPS * 1.6)
        for i, ((label, _), ep) in enumerate(zip(picks, eps)):
            r, c = divmod(i, 3)
            x, y = x0 + c * (tw + gx), y0 + r * (th + gy)
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            lab, col = outcome(ep)
            paste_frame(img, ep['frames'][k], (x, y, tw, th), border=col if done else PANEL)
            text(d, (x, y - 26), label, 19, AMBER, bold=True)
            if done:
                text(d, (x + tw, y - 24), f'{lab}  {ep["total"]:+.0f}', 17, col, anchor='ra')
        return img
    longest = max(len(e['frames']) for e in eps) / (FPS * 1.6)
    return Scene(name, render, [5.0, max(8.0, longest + 1.5)], min_beats=[0, longest + 1.0])


# ----------------------------------------------------------------------------
# Part 4: the weights, carefully
# ----------------------------------------------------------------------------

def pick_moments(ctx):
    """Moments of the showcase flight that each show one rule at work."""
    S = np.asarray(ctx.best_ep['states'])
    A = np.asarray(ctx.best_ep['actions'])
    air = (S[:, 6] == 0) & (S[:, 7] == 0)
    out = {}
    pick = {'falling': (air & (A == 2), S[:, 3]),                    # main engine, falling fastest
            'tilted': (air & (A == 1), S[:, 4]),                     # left engine, tilted furthest right
            'spinning': (air & (A == 3), -S[:, 5]),                  # right engine, spinning left fastest
            'landed': ((S[:, 6] == 1) & (S[:, 7] == 1), np.arange(len(S)))}
    for k, (mask, key) in pick.items():
        if mask.any():
            out[k] = int(np.argmin(np.where(mask, key, np.inf)))
    return out


def scene_worked(ctx, name='w_worked', moments=('falling', 'tilted')):
    """One beat per moment: each input × its weight, summed into four scores; argmax wins."""
    Wm, bias = ctx.best.weights, ctx.best.bias
    found = pick_moments(ctx)
    steps = [found[m] for m in moments if m in found]
    frames, states = ctx.best_ep['frames'], ctx.best_ep['states']
    gx, gy, lw, cw, rh = 500, 160, 110, 165, 40

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'One decision, step by step', t, 'input × weight, added up per action')
        k, lt, dur = b.local(t)
        step = steps[min(k, len(steps) - 1)]
        s = states[step]
        paste_frame(img, frames[step], (40, 140, 420, 280))
        text(d, (40, 435), 'the inputs right now', 18, BLUE, bold=True)
        for i in range(8):
            y = 464 + i * 26
            text(d, (40, y), STATE_SHORT[i], 17, FG)
            text(d, (200, y), f'{s[i]:+.2f}', 17, BLUE, mono=True)
            text(d, (270, y), (STATE_MEANING[i][0] if s[i] > 0 else (STATE_MEANING[i][1] or ''))
                 if abs(s[i]) > 0.05 else '', 14, MUTED)
        contrib = s[:, None] * Wm
        scores = contrib.sum(axis=0) + bias
        win = int(np.argmax(scores))
        cmax = max(1e-6, np.abs(contrib).max())
        reveal = (lt - 0.3) / max(0.5, dur * 0.45)
        for j in range(4):
            text(d, (gx + lw + j * cw + cw // 2, gy - 18), ACTION_NAMES[j], 17,
                 ACTION_COLORS[j] if j == win and reveal > 1.2 else MUTED, bold=True, anchor='mm')
        for i in range(9):
            y = gy + i * rh
            a = fade(reveal * 9, i, 1.0) if i < 8 else fade(reveal * 9, 8, 1.0)
            text(d, (gx, y + 10), STATE_SHORT[i] if i < 8 else 'bias', 17, FG if i < 8 else MUTED, a)
            for j in range(4):
                x = gx + lw + j * cw
                v = contrib[i, j] if i < 8 else bias[j]
                cx = x + cw // 2
                d.line([cx, y + 2, cx, y + rh - 4], fill=PANEL, width=1)
                bw = int(v / cmax * (cw // 2 - 6) * a)
                d.rectangle([min(cx, cx + bw), y + 8, max(cx, cx + bw), y + rh - 10],
                            fill=AMBER if v > 0 else BLUE)
                if abs(v) > 0.05 * cmax:
                    text(d, (cx + (5 if v < 0 else -5), y + 11), f'{v:+.1f}', 13, FG,
                         a, mono=True, anchor='la' if v < 0 else 'ra')
        ys = gy + 9 * rh + 14
        a = fade(reveal, 1.05, 0.3)
        d.line([gx, ys - 6, gx + lw + 4 * cw, ys - 6], fill=MUTED, width=1)
        text(d, (gx, ys + 4), 'score', 19, AMBER, a, bold=True)
        for j in range(4):
            x = gx + lw + j * cw
            on = j == win and reveal > 1.25
            if on:
                d.rounded_rectangle([x + 6, ys - 2, x + cw - 6, ys + 34], 8, fill=mix(BG, ACTION_COLORS[j], 0.35))
            text(d, (x + cw // 2, ys + 16), f'{scores[j]:+.2f}', 20, FG, a, bold=on, mono=True, anchor='mm',
                 bg=mix(BG, ACTION_COLORS[j], 0.35) if on else BG)
        if reveal > 1.25:
            text(d, (gx + lw + 2 * cw, ys + 60), f'highest score → {ACTION_NAMES[win]}', 24,
                 ACTION_COLORS[win], fade(reveal, 1.25, 0.3), bold=True, anchor='mm')
        return img
    return Scene(name, render, [8.0] * len(steps), min_beats=[6.0] * len(steps))


def scene_scale(ctx, name='w_scale'):
    """Beats: raw weights; how much each input varies in flight; the product = influence."""
    Wm = ctx.best.weights
    std = np.std(ctx.airborne_states(), axis=0)
    inf = ctx.influence()
    top = [divmod(int(k), 4) for k in np.argsort(inf.ravel())[::-1][:3]]

    def small_map(d, M, x0, y0, title, a, col_fn, fmt, outline=()):
        text(d, (x0, y0 - 46), title, 20, FG, a, bold=True)
        for j in range(4):
            text(d, (x0 + j * 62 + 30, y0 - 14), ACTION_SHORT[j], 13, MUTED, a, anchor='mm')
        for i in range(8):
            for j in range(4):
                x, y = x0 + j * 62, y0 + i * 44
                c = mix(BG, col_fn(M[i, j]), a)
                d.rectangle([x, y, x + 58, y + 40], fill=c)
                ink = BG if sum(c) > 520 else FG
                text(d, (x + 29, y + 20), fmt(M[i, j]), 13, ink, a, mono=True, anchor='mm', bg=c)
                if (i, j) in outline:
                    d.rectangle([x - 3, y - 3, x + 61, y + 43], outline=FG, width=3)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Big weight ≠ big effect', t, 'what matters is weight × how much the input moves')
        Y0 = 210
        for i in range(8):
            text(d, (190, Y0 + 22 + i * 44), STATE_SHORT[i], 17, FG, anchor='rm')
        wmax = np.abs(Wm).max()
        small_map(d, np.abs(Wm), 210, Y0, '|weight|', fade(t, 0.2),
                  lambda v: mix(PANEL, FG, v / wmax), lambda v: f'{v:.1f}')
        a = fade(t, b.at(1))
        text(d, (510, Y0 + 170), '×', 44, MUTED, a, anchor='mm')
        text(d, (580, Y0 - 46), 'spread in flight', 20, FG, a, bold=True)
        smax = std[:6].max()
        for i in range(8 if a > 0 else 0):
            y = Y0 + i * 44
            w_ = int(std[i] / smax * 150 * a) if i < 6 else 0
            d.rectangle([580, y + 8, 580 + w_, y + 32], fill=BLUE)
            text(d, (740, y + 10), f'{std[i]:.2f}' if i < 6 else 'always 0 in flight', 15, MUTED, a)
        a = fade(t, b.at(2))
        text(d, (900, Y0 + 170), '=', 44, MUTED, a, anchor='mm')
        imax = inf.max()
        small_map(d, inf, 960, Y0, 'influence', a, lambda v: mix(PANEL, AMBER, v / imax),
                  lambda v: f'{v:.2f}', outline=top if t > b.at(2) + 1.5 else ())
        return img
    return Scene(name, render, [6.0, 6.0, 7.0])


def scene_policy_map(ctx, name='w_policy_map', fx=4, fy=5, rx=(-0.5, 0.5), ry=(-0.6, 0.6),
                     fixed=None, subtitle='tilt vs. spin, everything else held steady'):
    """Beats: the decision map over two inputs; the real flight traced on top."""
    S = ctx.airborne_states()
    base = np.median(S, axis=0) if fixed is None else np.asarray(fixed, dtype=float)
    base[6:] = 0
    n = 120
    gxs = np.linspace(rx[0], rx[1], n)
    gys = np.linspace(ry[1], ry[0], n)
    grid = np.repeat(base[None, :], n * n, axis=0)
    grid[:, fx] = np.tile(gxs, n)
    grid[:, fy] = np.repeat(gys, n)
    acts = np.argmax(grid @ ctx.best.weights + ctx.best.bias, axis=1).reshape(n, n)
    pal = np.array([mix(BG, c, 0.55) for c in ACTION_COLORS], dtype=np.uint8)
    mx0, my0, mw, mh = 120, 130, 540, 500
    mapimg = Image.fromarray(pal[acts]).resize((mw, mh), Image.NEAREST)
    trace = np.asarray(ctx.best_ep['states'])
    tr_act = ctx.best_ep['actions']

    def to_px(vx, vy):
        return (mx0 + (vx - rx[0]) / (rx[1] - rx[0]) * mw, my0 + (ry[1] - vy) / (ry[1] - ry[0]) * mh)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The whole rule in one picture', t, subtitle)
        img.paste(mapimg, (mx0, my0))
        d.rectangle([mx0, my0, mx0 + mw, my0 + mh], outline=PANEL, width=2)
        cx, cy = to_px(0, 0)
        if rx[0] < 0 < rx[1]:
            d.line([cx, my0, cx, my0 + mh], fill=mix(BG, FG, 0.3))
        if ry[0] < 0 < ry[1]:
            d.line([mx0, cy, mx0 + mw, cy], fill=mix(BG, FG, 0.3))
        text(d, (mx0 + mw / 2, my0 + mh + 14), f'{STATE_NAMES[fx]}  →', 18, FG, anchor='ma')
        text(d, (mx0 + mw / 2, my0 + mh + 40), f'{rx[0]:+.1f}  …  {rx[1]:+.1f}', 14, MUTED, anchor='ma')
        text(d, (mx0 - 12, my0 + mh / 2), STATE_SHORT[fy], 18, FG, anchor='rm')
        text(d, (mx0 - 12, my0 + 12), f'{ry[1]:+.1f}', 14, MUTED, anchor='rm')
        text(d, (mx0 - 12, my0 + mh - 12), f'{ry[0]:+.1f}', 14, MUTED, anchor='rm')
        x = 720
        text(d, (x, 140), 'colour = the action it picks', 20, FG, bold=True)
        for j in range(4):
            y = 180 + j * 38
            d.rectangle([x, y, x + 28, y + 28], fill=pal[j].tolist() and tuple(int(v) for v in pal[j]))
            share = (acts == j).mean() * 100
            text(d, (x + 42, y + 3), f'{ACTION_NAMES[j]}', 20, FG)
            text(d, (x + 330, y + 3), f'{share:.0f}% of map', 16, MUTED, anchor='ra')
        text(d, (x, 350), 'other inputs held at a typical', 16, MUTED)
        text(d, (x, 372), 'in-flight value:', 16, MUTED)
        k = 0
        for i in range(6):
            if i in (fx, fy):
                continue
            text(d, (x + (k % 2) * 180, 400 + (k // 2) * 26), f'{STATE_SHORT[i]} = {base[i]:+.2f}', 16,
                 MUTED, mono=True)
            k += 1
        if b.index(t) >= 1:
            u = (t - b.at(1)) / max(1.0, b.durations[1] - 1.0)
            upto = max(2, int(min(1.0, u) * len(trace)))
            pts = [to_px(np.clip(s[fx], *rx), np.clip(s[fy], *ry)) for s in trace[:upto]
                   if s[6] == 0 and s[7] == 0]
            if len(pts) > 1:
                d.line(pts, fill=FG, width=2)
                hx, hy = pts[-1]
                col = ACTION_COLORS[tr_act[min(upto - 1, len(tr_act) - 1)]]
                d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=col, outline=FG, width=2)
            a = fade(t, b.at(1))
            text(d, (x, 500), 'white line: the real flight', 18, FG, a, bold=True)
            text(d, (x, 526), '(dot colour = what it actually did)', 15, MUTED, a)
            paste_frame(img, ctx.best_ep['frames'][min(upto - 1, len(ctx.best_ep['frames']) - 1)],
                        (x, 560, 165, 110))
        return img
    return Scene(name, render, [7.0, 7.0], min_beats=[0, 6.0])


def scene_story(ctx, rules, name='w_story'):
    """One beat per rule: `rules` = [(action j, [(i, j) cells to outline])]."""
    Wm, bias = ctx.best.weights, ctx.best.bias
    vmax = np.abs(Wm).max()
    A = np.asarray(ctx.best_ep['actions'])
    frames = ctx.best_ep['frames']
    x0, y0, cw, ch = 330, 150, 120, 46

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'What it learned, one action at a time', t)
        k = b.index(t)
        j, cells = rules[min(k, len(rules) - 1)]
        dim = lambda i, jj: 0.0 if (i, jj) in cells else (0.45 if jj == j else 0.85)  # noqa: E731
        heatmap(d, Wm, bias, x0, y0, cw, ch, vmax, dim=dim, outline=cells)
        x = 880
        text(d, (x, 150), ACTION_NAMES[j], 30, ACTION_COLORS[j], bold=True)
        idx = np.where(A == j)[0]
        if len(idx):
            f = frames[idx[int((t - b.at(k)) * FPS / 3) % len(idx)]]
            paste_frame(img, f, (x, 200, 345, 230), border=ACTION_COLORS[j])
            text(d, (x, 444), f'it fires {len(idx) / len(A) * 100:.0f}% of the time', 18, MUTED)
        for n, (i, jj) in enumerate(cells):
            w = bias[jj] if i == 8 else Wm[i, jj]
            if i == 8:
                s = f'bias {w:+.1f}: a constant {"head start" if w > 0 else "handicap"}'
            elif jj != j:
                side = STATE_MEANING[i][1 if w > 0 else 0] or STATE_MEANING[i][0]
                s = f'{STATE_SHORT[i]} → {ACTION_SHORT[jj]} {w:+.1f}: when {side}, {ACTION_SHORT[jj]} loses votes'
            elif w > 0 or STATE_MEANING[i][1]:
                s = f'{STATE_SHORT[i]} {w:+.1f}: votes for it when {STATE_MEANING[i][0 if w > 0 else 1]}'
            else:
                s = f'{STATE_SHORT[i]} {w:+.1f}: votes against it when {STATE_MEANING[i][0]}'
            text(d, (x, 486 + 28 * n), s, 15, FG, fade(t, b.at(k) + 0.6 + 0.4 * n))
        return img
    return Scene(name, render, [7.0] * len(rules))


def scene_ablation(ctx, name='w_ablation'):
    """Beats: zero each weight, measure landings; fly the worst damaged pilot."""
    ab = ctx.ablation()
    drop = (ab['full'] - ab['rates']) * 100
    worst = int(np.argmax(drop[:32]))
    wi, wj = divmod(worst, 4)
    p = ctx.best.get_parameters().copy()
    p[worst] = 0.0
    broken = record(controller_from(p), seed=ctx.seed)
    x0, y0, cw, ch = 330, 160, 110, 44
    dmax = max(1.0, drop.max())

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Which weights does it need?', t, 'set one weight to zero, fly 30 episodes')
        n = int(36 * ease((t - 0.4) / max(1.0, b.durations[0] - 1.5)))
        for j in range(4):
            text(d, (x0 + j * cw + cw // 2, y0 - 20), ACTION_SHORT[j], 16, MUTED, anchor='mm')
        for k in range(36):
            i, j = divmod(k, 4)
            y = y0 + i * ch + (10 if i == 8 else 0)
            if j == 0:
                text(d, (x0 - 14, y + ch // 2), STATE_NAMES[i] if i < 8 else 'bias', 17,
                     FG if i < 8 else MUTED, anchor='rm')
            on = k < n
            v = drop[k]
            col = mix(PANEL, RED, max(0.0, v) / dmax) if on else PANEL
            d.rectangle([x0 + j * cw, y, x0 + j * cw + cw - 4, y + ch - 4], fill=col)
            if on:
                text(d, (x0 + j * cw + cw // 2 - 2, y + ch // 2 - 2), f'−{v:.0f}%' if v > 0.5 else '—',
                     15, FG, mono=True, anchor='mm', bg=col)
            if k == worst and b.index(t) >= 1:
                d.rectangle([x0 + j * cw - 3, y - 3, x0 + j * cw + cw, y + ch], outline=FG, width=3)
        x = 820
        text(d, (x, 150), f'all weights: {ab["full"] * 100:.0f}% land', 22, GREEN, bold=True)
        text(d, (x, 186), 'number = drop in landing rate', 17, MUTED)
        if b.index(t) >= 1:
            a = fade(t, b.at(1))
            text(d, (x, 250), f'without {STATE_SHORT[wi]} → {ACTION_SHORT[wj]}:', 22, FG, a, bold=True)
            text(d, (x, 282), f'{ab["rates"][worst] * 100:.0f}% land', 22, RED, a, bold=True)
            play = max(0.0, t - b.at(1) - 0.3)
            fr = broken['frames']
            k = min(int(play * FPS), len(fr) - 1)
            paste_frame(img, fr[k], (x, 330, 400, 267), border=RED if k == len(fr) - 1 else PANEL)
            if k == len(fr) - 1:
                lab, col = outcome(broken)
                text(d, (x, 610), lab, 22, col, bold=True)
        return img
    fl = len(broken['frames']) / FPS
    return Scene(name, render, [8.0, max(7.0, fl + 1.5)], min_beats=[6.0, fl + 1.0])


def scene_outro(ctx, name='outro'):
    stats = ctx.stats
    items = [(f'{stats["landing_rate"]:.0f}%', 'landing rate', GREEN),
             (f'{stats["avg_reward"]:.0f}', 'average reward', AMBER),
             (f'{stats["avg_steps"]:.0f}', 'steps per episode', BLUE),
             ('36', 'parameters', FG)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Results', t, '100 new episodes, never seen in training')
        for i, (v, lab, col) in enumerate(items):
            x = 190 + i * 300
            a = fade(t, 0.5 + 0.3 * i)
            text(d, (x, 260), v, 72, col, a, bold=True, anchor='mm')
            text(d, (x, 320), lab, 20, MUTED, a, anchor='mm')
        a = fade(t, b.at(1))
        text(d, (W // 2, 450), 'github.com/JCA374/Moonlander-Genetic', 34, FG, a, bold=True, anchor='mm')
        text(d, (W // 2, 505), 'pip install -r requirements.txt  &&  python train_ga.py --parallel 4',
             20, MUTED, a, mono=True, anchor='mm')
        return img
    return Scene(name, render, [6.0, 6.0])
