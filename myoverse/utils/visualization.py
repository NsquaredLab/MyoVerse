import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from matplotlib import colormaps

from myoverse.datatypes import KinematicsData


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form numlines x (points per line) x 2 (x and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def jitter_points(
    x, y, spread_factor=0.9, seed=None, bw_method=None, one_sided=False
) -> np.ndarray:
    """Jitters points along the x-axis.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the points to jitter.
    y : np.ndarray
        The y-coordinates of the points to jitter.
    spread_factor : float, optional
        The spread factor of the jitter. The default is 0.9. A spread factor of 1.0 means that the jitter is as big as the
        distance between the x-coordinates of the points.
    seed : int, optional
        The seed to use for the random number generator. The default is None.
    bw_method : str, optional
        The bandwidth method to use for the kernel density estimation. The default is None. See the documentation of
        scipy.stats.gaussian_kde for more information.
    one_sided : bool, optional
        Whether to only jitter the points to the right of the original x-coordinates. The default is False.
        If True only positive jitter is applied.

    Returns
    -------
    np.ndarray
        The jittered x-coordinates.
    """
    # Set seed for reproducibility
    # Set seed for reproducibility
    np.random.seed(seed)

    x_copied = np.copy(x)
    y_copied = np.copy(y)

    x_jittered = []

    _, idx = np.unique(x_copied, return_index=True)
    for unique_x in x_copied[np.sort(idx)]:
        x_matches = np.where(x_copied == unique_x)[0]

        x_old = x_copied[x_matches].copy()
        y_old = y_copied[x_matches].copy()

        kde = gaussian_kde(y_old, bw_method=bw_method)
        density = kde(y_old)

        weights = density / np.max(np.abs(density))

        if one_sided:
            spread = (
                np.random.uniform(low=0, high=spread_factor, size=len(y_old)) * weights
            )
        else:
            spread = (
                np.random.uniform(
                    low=-spread_factor, high=spread_factor, size=len(y_old)
                )
                * weights
            )

        x_jittered.append(x_old + spread)

    return np.concatenate(x_jittered)


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("cool"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3.0,
    alpha=1.0,
    ax=None,
    capstyle="round",
):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        antialiaseds=True,
        capstyle=capstyle,
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def plot_predicted_and_ground_truth_kinematics(
    predictions: KinematicsData,
    ground_truths: KinematicsData,
    prediction_representation: str,
    ground_truth_representation: str,
    wrist_included: bool = True,
    nr_of_fingers: int = 5,
    background_style: str = 'seaborn-v0_8-whitegrid',  # Alternative: 'default' for white background
):
    """
    Plot the predicted and ground truth kinematics interactively.

    Parameters
    ----------
    predictions : KinematicsData
        The predicted kinematics.
    ground_truths : KinematicsData
        The ground truth kinematics.
    prediction_representation : str
        The representation of the predicted kinematics.
    ground_truth_representation : str
        The representation of the ground truth kinematics.
    wrist_included : bool, optional
        Whether the wrist is included in the kinematics. The default is True.
    nr_of_fingers : int, optional
        Number of fingers in the hand model. The default is 5.
    background_style : str, optional
        Matplotlib style to use. The default is 'seaborn-v0_8-whitegrid'.
    """
    if prediction_representation not in predictions.processed_representations.keys():
        raise KeyError(
            f'The representation "{prediction_representation}" does not exist.'
        )

    if (
        ground_truth_representation
        not in ground_truths.processed_representations.keys()
    ):
        raise KeyError(
            f'The representation "{ground_truth_representation}" does not exist.'
        )

    prediction_kinematics = predictions[prediction_representation]
    ground_truth_kinematics = ground_truths[ground_truth_representation]

    if not wrist_included:
        if prediction_kinematics.ndim == 3 and prediction_kinematics.shape[1] == 3:
             prediction_kinematics = np.concatenate(
                [np.zeros((1, 3, prediction_kinematics.shape[2])), prediction_kinematics],
                axis=0,
            )
        else:
             raise ValueError("Unexpected shape for prediction_kinematics when wrist_included=False")

        if ground_truth_kinematics.ndim == 3 and ground_truth_kinematics.shape[1] == 3:
            ground_truth_kinematics = np.concatenate(
                [
                    np.zeros((1, 3, ground_truth_kinematics.shape[2])),
                    ground_truth_kinematics,
                ],
                axis=0,
            )
        else:
            raise ValueError("Unexpected shape for ground_truth_kinematics when wrist_included=False")

    # Apply selected style - 'seaborn-v0_8-whitegrid' gives a cleaner look than 'darkgrid'
    plt.style.use(background_style)
    
    fig = plt.figure(figsize=(14, 7))  # Slightly larger figure
    fig.suptitle(f"Kinematics Comparison\nPrediction: '{prediction_representation}' vs Ground Truth: '{ground_truth_representation}'", 
                fontsize=16, y=0.98)  # Adjusted position

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    ax1.set_title("Predicted Kinematics", fontsize=14, pad=10)
    ax2.set_title("Ground Truth Kinematics", fontsize=14, pad=10)

    all_kinematics = np.concatenate([prediction_kinematics, ground_truth_kinematics], axis=2)
    min_coords = all_kinematics.min(axis=(0, 2))
    max_coords = all_kinematics.max(axis=(0, 2))
    mid_coords = (max_coords + min_coords) / 2.0
    max_range = (max_coords - min_coords).max() * 1.1

    axes = [ax1, ax2]
    for ax in axes:
        ax.set_xlim(mid_coords[0] - max_range / 2.0, mid_coords[0] + max_range / 2.0)
        ax.set_ylim(mid_coords[1] - max_range / 2.0, mid_coords[1] + max_range / 2.0)
        ax.set_zlim(mid_coords[2] - max_range / 2.0, mid_coords[2] + max_range / 2.0)

        ax.set_aspect('equal', adjustable='box')

        # Make grid lines lighter and thinner
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Set better viewing angle for hand visualization
        ax.view_init(elev=20, azim=-60)  # Adjust these values for best visualization
        
        # Improve axis labels with units if applicable (assuming mm)
        ax.set_xlabel("X (mm)", fontsize=12)
        ax.set_ylabel("Y (mm)", fontsize=12)
        ax.set_zlabel("Z (mm)", fontsize=12)

    # Improve joint visualization
    initial_sample_idx = 0
    joint_color = 'black'
    joint_marker = 'o'
    joint_size = 12  # Reduced from 20 to 12
    alpha_joints = 0.9  # More opaque
    alpha_lines = 0.8  # Slightly transparent lines

    cmap = colormaps.get_cmap('tab10')
    finger_colors = [cmap(i) for i in range(nr_of_fingers)]

    # Plot wrist joint with a different color/size to highlight it
    wrist_size = 18  # Reduced from 30 to 18
    
    # Plot all joints
    (prediction_joints_plot,) = ax1.plot(
        *prediction_kinematics[..., initial_sample_idx].T,
        marker=joint_marker, color=joint_color, linestyle='', 
        markersize=joint_size, alpha=alpha_joints
    )
    (ground_truth_joints_plot,) = ax2.plot(
        *ground_truth_kinematics[..., initial_sample_idx].T,
        marker=joint_marker, color=joint_color, linestyle='', 
        markersize=joint_size, alpha=alpha_joints
    )
    
    # Add separate wrist markers (optional)
    (prediction_wrist_plot,) = ax1.plot(
        *prediction_kinematics[0:1, :, initial_sample_idx].T,
        marker='o', color='darkblue', linestyle='', 
        markersize=wrist_size, alpha=alpha_joints
    )
    (ground_truth_wrist_plot,) = ax2.plot(
        *ground_truth_kinematics[0:1, :, initial_sample_idx].T,
        marker='o', color='darkblue', linestyle='', 
        markersize=wrist_size, alpha=alpha_joints
    )

    prediction_finger_plots = []
    ground_truth_finger_plots = []
    for finger in range(nr_of_fingers):
        finger_indices = [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))

        prediction_finger_plots.append(
            ax1.plot(
                *prediction_kinematics[finger_indices, :, initial_sample_idx].T,
                color=finger_colors[finger],
                linewidth=3.0,  # Thicker lines
                alpha=alpha_lines
            )[0]
        )
        ground_truth_finger_plots.append(
            ax2.plot(
                *ground_truth_kinematics[finger_indices, :, initial_sample_idx].T,
                color=finger_colors[finger],
                linewidth=3.0,  # Thicker lines
                alpha=alpha_lines
            )[0]
        )

    # Add a legend for fingers (optional)
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'][:nr_of_fingers]
    # Create dummy plots for legend
    dummy_lines = []
    for i in range(nr_of_fingers):
        dummy_lines.append(plt.Line2D([0], [0], color=finger_colors[i], lw=3))
    
    # Add legend to figure (outside plots)
    fig.legend(dummy_lines, finger_names, loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), ncol=nr_of_fingers, fontsize=12)

    # Improve slider appearance
    slider_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])  # Moved up slightly to make room for legend
    slider_color = 'steelblue'  # More attractive color
    sample_slider = Slider(
        ax=slider_ax,
        label="Sample Index",
        valmin=0,
        valmax=prediction_kinematics.shape[2] - 1,
        valstep=1,
        valinit=initial_sample_idx,
        color=slider_color
    )

    def update(val):
        current_sample_idx = int(val)
        prediction_kinematics_new_sample = prediction_kinematics[..., current_sample_idx]
        ground_truth_kinematics_new_sample = ground_truth_kinematics[..., current_sample_idx]

        # Update joint positions
        prediction_joints_plot.set_data_3d(*prediction_kinematics_new_sample.T)
        ground_truth_joints_plot.set_data_3d(*ground_truth_kinematics_new_sample.T)
        
        # Update wrist positions
        prediction_wrist_plot.set_data_3d(*prediction_kinematics[0:1, :, current_sample_idx].T)
        ground_truth_wrist_plot.set_data_3d(*ground_truth_kinematics[0:1, :, current_sample_idx].T)

        # Update finger lines
        for finger in range(nr_of_fingers):
            finger_indices = [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))

            prediction_finger_plots[finger].set_data_3d(
                *prediction_kinematics[finger_indices, :, current_sample_idx].T
            )
            ground_truth_finger_plots[finger].set_data_3d(
                *ground_truth_kinematics[finger_indices, :, current_sample_idx].T
            )
        
        fig.canvas.draw_idle()

    sample_slider.on_changed(update)
    
    # Add sample count display
    total_samples = prediction_kinematics.shape[2]
    fig.text(0.5, 0.12, f"Total Samples: {total_samples}", 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])  # Adjusted to make room for finger legend

    plt.show()
