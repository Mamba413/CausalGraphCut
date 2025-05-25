import matplotlib.pyplot as plt
import numpy as np

def plot_hexagon_internal(ax, center, size, color, label=None):
    """Plot a single hexagon centered at 'center' with given 'size' and 'color'. Optionally label the hexagon."""
    angle = np.linspace(0, 2 * np.pi, 7)
    x_hexagon = center[0] + size * np.cos(angle)
    y_hexagon = center[1] + size * np.sin(angle)
    ax.fill(x_hexagon, y_hexagon, edgecolor='#4d4d4d', facecolor=color, lw=2)
    if label is not None:
        ax.text(center[0], center[1], label, ha='center', va='center', fontsize=12, color='black')

def plot_hexagon(coords, cluster):
    # Plotting the hexagonal grid with index numbers
    fig, ax = plt.subplots(figsize=(10, 8))

    for (x_center, y_center), c_label in zip(coords, cluster):
        color = plt.cm.tab20(c_label)  # Color based on cluster number
        plot_hexagon_internal(ax, (x_center, y_center), 1, color, label=f"{c_label}")

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Turn off axis labels and ticks
    plt.title('Hexagonal Grid with Index Visualization')
    plt.show()

