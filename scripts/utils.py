#!/usr/bin/env python
"""
Shared helpers for the scripts/ diagnostics.

Two things live here:

1. Loss curves. `LossHistory` accumulates the per-step `output['losses']` dict
   that AniMerPlusPlus.compute_loss returns, and `plot_loss_history` writes one
   PNG per loss term plus a single grid figure holding all of them. The loss
   keys are whatever compute_loss emitted (they are built dynamically, see
   animerpp.py:316-342), so nothing here hardcodes a list of loss names -- the
   combined 'loss' key is just another series and gets its own plot like the
   rest.

2. Labelled render grids. `model.tensorboard_logging` returns a
   `make_grid(..., nrow=5)` tensor whose five columns are always
   GRID_COLUMN_LABELS, but the raw grid is unlabelled and the columns are easy
   to mix up. `save_labeled_grid` writes it out with a header strip naming each
   column.
"""
import json
import os

import numpy as np

# Column order of the grid built by MeshRenderer.visualize_tensorboard
# (mesh_renderer.py:221-229) and laid out by AniMerPlusPlus.tensorboard_logging
# with nrow=5 (animerpp.py:394). Keep in sync with those if the tiles change.
GRID_COLUMN_LABELS = ("Input", "Mesh Overlay", "Mesh View",
                      "2D Keypoints Overlay", "2D Keypoints GT")
GRID_NCOL = 5
GRID_PADDING = 2

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def loss_value(v):
    """Loss entries are detached 0-dim tensors; floats/numpy scalars pass through."""
    return float(v.item()) if hasattr(v, 'item') else float(v)


class LossHistory:
    """Per-step record of every loss term, keyed by loss name.

    Terms are tracked independently (each with its own step list) so a loss that
    only appears from some step onwards, or is missing from a step, plots
    correctly instead of shifting the whole series.
    """

    def __init__(self):
        self.steps = {}
        self.values = {}

    def record(self, step, losses):
        for name, v in losses.items():
            self.steps.setdefault(name, []).append(int(step))
            self.values.setdefault(name, []).append(loss_value(v))

    def keys(self):
        """Loss names, with the combined 'loss' first so it leads every figure."""
        names = sorted(self.values)
        if 'loss' in names:
            names.remove('loss')
            names.insert(0, 'loss')
        return names

    def is_empty(self):
        return not self.values

    def recorded_steps(self):
        """Distinct step numbers held, across all terms."""
        return sorted({step for steps in self.steps.values() for step in steps})

    def truncate_from(self, step):
        """Drop everything recorded at or after `step`."""
        for name in list(self.values):
            kept = [i for i, s in enumerate(self.steps[name]) if s < step]
            self.steps[name] = [self.steps[name][i] for i in kept]
            self.values[name] = [self.values[name][i] for i in kept]
            if not self.steps[name]:
                del self.steps[name]
                del self.values[name]

    def save_json(self, path):
        """Raw history, so a run can be replotted without retraining."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {name: {'steps': self.steps[name], 'values': self.values[name]}
                   for name in self.keys()}
        with open(path, 'w') as f:
            json.dump(payload, f)
        return path


def load_loss_history(path):
    """Inverse of LossHistory.save_json."""
    with open(path) as f:
        payload = json.load(f)
    history = LossHistory()
    history.steps = {k: v['steps'] for k, v in payload.items()}
    history.values = {k: v['values'] for k, v in payload.items()}
    return history


def resume_loss_history(path, start_step):
    """Continue an earlier run's curves instead of restarting them at the resume
    step, so a paused-and-resumed run still plots as one history.

    Returns everything recorded before `start_step`, or an empty history if
    there is no file to resume from. Entries at or after `start_step` are
    dropped: resuming from an older checkpoint than the last one recorded means
    those steps are about to be retrained, and keeping both would put two
    different values on the same step.
    """
    if not path or not os.path.exists(path):
        return LossHistory()
    history = load_loss_history(path)
    history.truncate_from(start_step)
    return history


def rolling_mean(values, window):
    """Trend line for the raw curve. Minibatch SGD makes per-step losses noisy
    enough that the shape is hard to read; the raw series is still plotted
    underneath so nothing is hidden by the smoothing."""
    if window <= 1 or len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode='valid')


def _plot_one_loss(ax, name, steps, values, smooth_window):
    """Draw a single loss term onto an existing axes."""
    ax.plot(steps, values, linewidth=1.0, alpha=0.45 if smooth_window > 1 else 1.0,
            color='tab:blue', label='per step')
    trend = rolling_mean(values, smooth_window)
    if trend is not None:
        ax.plot(steps[smooth_window - 1:], trend, linewidth=1.8, color='tab:red',
                label=f'rolling mean ({smooth_window})')
        ax.legend(fontsize=7, loc='upper right')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('step', fontsize=8)
    ax.set_ylabel('loss', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    positive = [v for v in values if v > 0]
    # Loss terms span several orders of magnitude over a run; a log axis is the
    # only way the tail is readable. Needs strictly-positive data.
    if positive and len(positive) == len(values) and max(values) / min(positive) > 50:
        ax.set_yscale('log')


def plot_loss_history(history, out_dir, smooth_window=25, prefix='', dpi=120):
    """One PNG per loss term (including the combined 'loss'), plus one grid
    figure holding every term. Returns the list of paths written."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if history.is_empty():
        return []
    os.makedirs(out_dir, exist_ok=True)
    names = history.keys()
    written = []

    for name in names:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_one_loss(ax, name, history.steps[name], history.values[name], smooth_window)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{prefix}{name}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.6 * nrow), squeeze=False)
    flat_axes = axes.ravel()
    for ax, name in zip(flat_axes, names):
        _plot_one_loss(ax, name, history.steps[name], history.values[name], smooth_window)
    for ax in flat_axes[len(names):]:
        ax.axis('off')
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}all_losses.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)
    return written


def _load_font(size):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    try:
        import matplotlib
        bundled = os.path.join(os.path.dirname(matplotlib.__file__),
                               'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')
        if os.path.exists(bundled):
            return ImageFont.truetype(bundled, size)
    except ImportError:
        pass
    return ImageFont.load_default()


def _fit_font(draw, labels, tile_w, max_size=22, min_size=7):
    """Largest font size at which every label still fits inside one tile."""
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(size)
        widths = [draw.textbbox((0, 0), text, font=font)[2] for text in labels]
        if max(widths) <= tile_w * 0.96:
            return font
    return _load_font(min_size)


def grid_to_pil(grid):
    """make_grid output (C, H, W) float in [0, 1] -> RGB PIL image."""
    from PIL import Image

    arr = grid.detach().float().cpu().clamp(0, 1).numpy()
    return Image.fromarray((np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8))


def save_labeled_grid(grid, out_path, labels=GRID_COLUMN_LABELS, ncol=GRID_NCOL,
                      padding=GRID_PADDING, bg=(255, 255, 255), fg=(0, 0, 0)):
    """Save a make_grid tensor with a header strip naming each column.

    Column geometry is derived from make_grid's own layout: it emits a grid of
    width `ncol * (tile_w + padding) + padding` with tile j starting at
    `j * (tile_w + padding) + padding`, so the tile width can be recovered from
    the image width alone and no tile size has to be passed in.
    """
    from PIL import Image, ImageDraw

    image = grid_to_pil(grid)
    labels = list(labels)[:ncol]
    tile_w = (image.width - padding) / ncol - padding
    if tile_w <= 0:
        image.save(out_path)
        return out_path

    probe = ImageDraw.Draw(image)
    font = _fit_font(probe, labels, tile_w)
    text_h = max(probe.textbbox((0, 0), text, font=font)[3] for text in labels)
    header_h = int(text_h + 10)

    canvas = Image.new('RGB', (image.width, image.height + header_h), bg)
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    for j, text in enumerate(labels):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        centre = j * (tile_w + padding) + padding + tile_w / 2
        draw.text((centre - (right - left) / 2 - left, (header_h - (bottom - top)) / 2 - top),
                  text, fill=fg, font=font)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    return out_path
