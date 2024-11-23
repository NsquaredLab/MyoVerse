import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde

from doc_octopy.datatypes import KinematicsData


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
):
    """
    Plot the predicted and ground truth kinematics.

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
        prediction_kinematics = np.concatenate(
            [np.zeros((1, 3, prediction_kinematics.shape[2])), prediction_kinematics],
            axis=0,
        )
        ground_truth_kinematics = np.concatenate(
            [
                np.zeros((1, 3, ground_truth_kinematics.shape[2])),
                ground_truth_kinematics,
            ],
            axis=0,
        )

    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(211, projection="3d")

    ax1.set_title("Predicted kinematics")
    ax2.set_title("Ground truth kinematics")

    # get biggest axis range
    max_range_ax1 = (
        np.array(
            [
                prediction_kinematics[:, 0].max() - prediction_kinematics[:, 0].min(),
                prediction_kinematics[:, 1].max() - prediction_kinematics[:, 1].min(),
                prediction_kinematics[:, 2].max() - prediction_kinematics[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    max_range_ax2 = (
        np.array(
            [
                ground_truth_kinematics[:, 0].max()
                - ground_truth_kinematics[:, 0].min(),
                ground_truth_kinematics[:, 1].max()
                - ground_truth_kinematics[:, 1].min(),
                ground_truth_kinematics[:, 2].max()
                - ground_truth_kinematics[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    ax1.set_xlim(
        prediction_kinematics[:, 0].mean() - max_range_ax1,
        prediction_kinematics[:, 0].mean() + max_range_ax1,
    )
    ax1.set_ylim(
        prediction_kinematics[:, 1].mean() - max_range_ax1,
        prediction_kinematics[:, 1].mean() + max_range_ax1,
    )
    ax1.set_zlim(
        prediction_kinematics[:, 2].mean() - max_range_ax1,
        prediction_kinematics[:, 2].mean() + max_range_ax1,
    )

    ax2.set_xlim(
        ground_truth_kinematics[:, 0].mean() - max_range_ax2,
        ground_truth_kinematics[:, 0].mean() + max_range_ax2,
    )
    ax2.set_ylim(
        ground_truth_kinematics[:, 1].mean() - max_range_ax2,
        ground_truth_kinematics[:, 1].mean() + max_range_ax2,
    )
    ax2.set_zlim(
        ground_truth_kinematics[:, 2].mean() - max_range_ax2,
        ground_truth_kinematics[:, 2].mean() + max_range_ax2,
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # create joint and finger plots
    (prediction_joints_plot,) = ax1.plot(
        *prediction_kinematics[..., 0].T, "o", color="black"
    )
    (ground_truth_joints_plot,) = ax2.plot(
        *ground_truth_kinematics[..., 0].T, "o", color="black"
    )

    prediction_figer_plots = []
    ground_truth_finger_plots = []
    for finger in range(nr_of_fingers):
        prediction_figer_plots.append(
            ax1.plot(
                *prediction_kinematics[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))), :, 0
                ].T,
                color="blue",
            )
        )
        ground_truth_finger_plots.append(
            ax2.plot(
                *ground_truth_kinematics[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))), :, 0
                ].T,
                color="blue",
            )
        )

    sample_slider = Slider(
        ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),
        label="Sample (a. u.)",
        valmin=0,
        valmax=prediction_kinematics.shape[2] - 1,
        valstep=1,
        valinit=0,
    )

    def update(val):
        prediction_kinematics_new_sample = prediction_kinematics[..., int(val)]
        ground_truth_kinematics_new_sample = ground_truth_kinematics[..., int(val)]

        prediction_joints_plot._verts3d = tuple(prediction_kinematics_new_sample.T)
        ground_truth_joints_plot._verts3d = tuple(ground_truth_kinematics_new_sample.T)

        for finger in range(nr_of_fingers):
            prediction_figer_plots[finger][0]._verts3d = tuple(
                prediction_kinematics[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                    :,
                    int(val),
                ].T
            )
            ground_truth_finger_plots[finger][0]._verts3d = tuple(
                ground_truth_kinematics[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                    :,
                    int(val),
                ].T
            )
        fig.canvas.draw_idle()

    sample_slider.on_changed(update)

    plt.show()
