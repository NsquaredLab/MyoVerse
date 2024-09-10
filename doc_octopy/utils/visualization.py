import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde


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


def draw_hand(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    nr_of_fingers: int,
    wrist_included: bool = True,
):
    """Draws a hand with the given predictions and ground truths.

    Parameters
    ----------
    predictions : np.ndarray
        The predictions to draw. Should be of shape (nr_of_samples, nr_of_joints, 3).
    ground_truths : np.ndarray
        The ground truths to draw. Should be of shape (nr_of_samples, nr_of_joints, 3).
    nr_of_fingers : int
        The number of fingers to draw.
    wrist_included : bool, optional
        Whether the wrist is included in the predictions and ground truths. The default is True.

    Raises
    ------
    ValueError
        If the number of joints is not 1 + 4 * nr_of_fingers
    """
    if not wrist_included:
        predictions = np.concatenate(
            [np.zeros((predictions.shape[0], 1, 3)), predictions], axis=1
        )
        ground_truths = np.concatenate(
            [np.zeros((ground_truths.shape[0], 1, 3)), ground_truths], axis=1
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # get biggest axis range
    max_range = (
        np.array(
            [
                ground_truths[..., 0].max() - ground_truths[..., 0].min(),
                ground_truths[..., 1].max() - ground_truths[..., 1].min(),
                ground_truths[..., 2].max() - ground_truths[..., 2].min(),
            ]
        ).max()
        / 2.0
    )

    # set axis limits
    ax.set_xlim(
        ground_truths[..., 0].mean() - max_range,
        ground_truths[..., 0].mean() + max_range,
    )
    ax.set_ylim(
        ground_truths[..., 1].mean() - max_range,
        ground_truths[..., 1].mean() + max_range,
    )
    ax.set_zlim(
        ground_truths[..., 2].mean() - max_range,
        ground_truths[..., 2].mean() + max_range,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # create joint and finger plots
    (joints_plot,) = ax.plot(*ground_truths[0].T, "o", color="black")

    finger_plots_prediction = []
    finger_plots_expectation = []
    for finger in range(nr_of_fingers):
        finger_plots_prediction.append(
            ax.plot(
                *predictions[
                    0, [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))
                ].T,
                color="red"
            )
        )
        finger_plots_expectation.append(
            ax.plot(
                *ground_truths[
                    0, [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))
                ].T,
                color="blue"
            )
        )

    sample_slider = Slider(
        ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),
        label="Sample (a. u.)",
        valmin=0,
        valmax=ground_truths.shape[0] - 1,
        valstep=1,
        valinit=0,
    )

    def update(val):
        outputs_new_sample = predictions[int(val)]
        ground_truths_new_sample = ground_truths[int(val)]

        joints_plot._verts3d = tuple(ground_truths_new_sample.T)

        for finger in range(nr_of_fingers):
            finger_plots_prediction[finger][0]._verts3d = tuple(
                outputs_new_sample[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))
                ].T
            )
            finger_plots_expectation[finger][0]._verts3d = tuple(
                ground_truths_new_sample[
                    [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4)))
                ].T
            )

        fig.canvas.draw_idle()

    sample_slider.on_changed(update)

    plt.show()
