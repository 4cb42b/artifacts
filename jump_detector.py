import pandas as pd
import numpy as np
from scipy.signal import medfilt
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def detect_step_edges(
    x,
    jump_up=True,         # True → detect jump; False → landing
    win=250, low_len=100, high_len=100,
    low_th=100, high_th=400,
    thr=50, high_clip=60,   # band for edge check
    min_spacing=100):
    """
    Detect one index per trapezoidal step edge in `x`.

    Parameters
    ----------
        x : 1-D array-like
        jump_up   : True  → detect the *start*  of a step (low→high)
                      False → detect the *finish* of a step (high→low)
    Logical convolutional kernel params: 
        win         : sliding-window length
        low_len     : # samples at start  of window
        high_len    : # samples at end   of window
    Post-convolutional midpoint detection:
        low_th, high_th : thresholds for the logical tests
        thr, high_clip  : band (thr … high_clip] used to flag an edge
        min_spacing : required gap between successive picks

    Returns
    -------
    idx : 1-D NumPy array of selected indices
    """

    if low_len + high_len > win:
        raise ValueError("low_len + high_len must be ≤ win")

    # ---- logical-kernel score -----------------------------------------------
    half   = win // 2
    w      = sliding_window_view(np.pad(x, half, mode="edge"), win)

    if jump_up:                           # feet of step: low→high
        left_bool  = w[:,  :low_len ] <  low_th
        right_bool = w[:, -high_len:] >  high_th
    else:                                   # end of step: high→low
        left_bool  = w[:,  :low_len ] >  high_th
        right_bool = w[:, -high_len:] <  low_th

    score = (left_bool * 2 - 1).sum(axis=1) + (right_bool * 2 - 1).sum(axis=1)
    s     = score

    # ---- rising vs. falling edge across the band ----------------------------
    if jump_up:
        edge_mask = ((s[1:] >= thr) & (s[1:] <= high_clip) &   # now inside band
                     (s[:-1] <  thr) &                         # was below band
                     (s[1:]  >  0))                            # positive trapezoid
    else:
        edge_mask = ((s[:-1] >= thr) & (s[:-1] <= high_clip) & # was inside band
                     (s[1:]  <  thr) &                         # now below band
                     (s[:-1] >  0))
    candidates = np.flatnonzero(edge_mask) + 1                 # first below thr

    # ---- one pick per trapezoid --------------------------------------------
    picks, last = [], -np.inf
    for i in candidates:
        if i - last >= min_spacing:
            picks.append(i)
            last = i

    return np.asarray(picks, dtype=int)


# 1) import data and take a slice
df = pd.read_csv('grfz_output.csv'); df.info()
df_slice = df[130000:140000]

# 2) pre-process raw signal
x      = df_slice.iloc[:, 1].to_numpy()
x_med  = medfilt(x, kernel_size=7)

# 3) detect feet and ends
edge_idx_start = detect_step_edges(x_med, jump_up=True)
edge_idx_end   = detect_step_edges(x_med, jump_up=False)

# 4) plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
ax.plot(x, linewidth=0.3)

ymin, ymax = ax.get_ylim()
ax.vlines(edge_idx_start, ymin, ymax, colors="crimson",
          linewidth=0.3)
ax.vlines(edge_idx_end,   ymin, ymax, colors="crimson",
          linewidth=0.3)

ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
