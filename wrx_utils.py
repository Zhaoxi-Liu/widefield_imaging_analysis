import matplotlib.pyplot as plt
import numpy as np

def plot_onset_index(frame_index, title=None):
    frame_interval = np.diff(onset_frame_index)
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(frame_interval)

    # set the y-axis range
    y_range = frame_interval.max() - frame_interval.min()
    y_min = frame_interval.min() - 3*y_range
    y_max = frame_interval.max() + 3*y_range
    ax.set_ylim(bottom=y_min, top=y_max)
    
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Frame interval')
    if title:
        plt.title(title)
    plt.show()