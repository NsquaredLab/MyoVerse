"""GPU-accelerated temporal transforms using PyTorch.

All transforms work with named tensors and run on any device (CPU, CUDA, MPS).

Filter implementations:
- Bandpass, Highpass, Lowpass: Use torchaudio IIR biquad filters for proper
  Butterworth-style roll-off. Multiple passes can be applied for steeper slopes.
- Notch: Uses FFT-based filtering for sharp, precise narrow-band removal
  (ideal for powerline interference at 50/60 Hz).

Example
-------
>>> import torch
>>> from myoverse.transforms import RMS, Bandpass, ZScore, Compose
>>>
>>> # Create EMG tensor on GPU
>>> emg = torch.randn(64, 20000, device='cuda', names=('channel', 'time'))
>>>
>>> # GPU-accelerated pipeline
>>> pipeline = Compose([
...     Bandpass(20, 450, fs=2048, dim='time'),
...     RMS(window_size=200, dim='time'),
...     ZScore(dim='time'),
... ])
>>> processed = pipeline(emg)  # All on GPU
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from myoverse.transforms.base import TensorTransform, get_dim_index


class RMS(TensorTransform):
    """Root Mean Square over sliding windows (GPU-accelerated).

    Uses unfold for efficient sliding window computation on GPU.

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows. If None, uses window_size (non-overlapping).
    dim : str
        Dimension to compute RMS over.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> rms = RMS(window_size=200, dim='time')
    >>> y = rms(x)  # Shape: (64, 10)
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        # Remove names for unfold (doesn't support named tensors)
        x = x.rename(None)

        # Unfold along time dimension: creates windows
        # Result shape: (..., n_windows, window_size)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x_unfolded ** 2, dim=-1))

        # Restore names (time dimension now represents windows)
        if names[0] is not None:
            rms = rms.rename(*names)

        return rms


class MAV(TensorTransform):
    """Mean Absolute Value over sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to compute MAV over.
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # MAV: mean(|x|)
        mav = torch.mean(torch.abs(x_unfolded), dim=-1)

        if names[0] is not None:
            mav = mav.rename(*names)

        return mav


class VAR(TensorTransform):
    """Variance over sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to compute variance over.
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # Variance
        var = torch.var(x_unfolded, dim=-1)

        if names[0] is not None:
            var = var.rename(*names)

        return var


class Rectify(TensorTransform):
    """Full-wave rectification (absolute value).

    Parameters
    ----------
    dim : str
        Dimension name (not used, but kept for API consistency).
    """

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class Bandpass(TensorTransform):
    """Bandpass filter using cascaded torchaudio biquads (GPU-accelerated).

    Uses cascaded highpass and lowpass IIR biquad filters for proper
    Butterworth-style roll-off at both cutoff frequencies.

    Parameters
    ----------
    low : float
        Low cutoff frequency in Hz.
    high : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (number of biquad passes per filter). Default 4.
    Q : float
        Quality factor. Default 0.707 for Butterworth response.
    dim : str
        Dimension to filter over.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> bp = Bandpass(20, 450, fs=2048, dim='time')
    >>> y = bp(x)
    """

    def __init__(
        self,
        low: float,
        high: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.low = low
        self.high = high
        self.fs = fs
        self.order = order
        self.Q = Q

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Move time dimension to last if not already
        if dim_idx != x.ndim - 1:
            x = x.movedim(dim_idx, -1)

        # Apply highpass at low cutoff (removes frequencies below low)
        for _ in range(self.order):
            x = AF.highpass_biquad(x, self.fs, self.low, self.Q)

        # Apply lowpass at high cutoff (removes frequencies above high)
        for _ in range(self.order):
            x = AF.lowpass_biquad(x, self.fs, self.high, self.Q)

        # Move time dimension back
        if dim_idx != x.ndim - 1:
            x = x.movedim(-1, dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x


class Highpass(TensorTransform):
    """Highpass filter using torchaudio biquad (GPU-accelerated).

    Uses IIR biquad filter with proper Butterworth-style roll-off.
    Can apply multiple passes for steeper slope.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (number of biquad passes). Default 4 for 24dB/oct slope.
    Q : float
        Quality factor. Default 0.707 for Butterworth response.
    dim : str
        Dimension to filter over.
    """

    def __init__(
        self,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.Q = Q

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Move time dimension to last if not already
        if dim_idx != x.ndim - 1:
            x = x.movedim(dim_idx, -1)

        # Apply biquad filter multiple times for higher order
        for _ in range(self.order):
            x = AF.highpass_biquad(x, self.fs, self.cutoff, self.Q)

        # Move time dimension back
        if dim_idx != x.ndim - 1:
            x = x.movedim(-1, dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x


class Lowpass(TensorTransform):
    """Lowpass filter using torchaudio biquad (GPU-accelerated).

    Uses IIR biquad filter with proper Butterworth-style roll-off.
    Can apply multiple passes for steeper slope.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (number of biquad passes). Default 4 for 24dB/oct slope.
    Q : float
        Quality factor. Default 0.707 for Butterworth response.
    dim : str
        Dimension to filter over.
    """

    def __init__(
        self,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.Q = Q

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Move time dimension to last if not already
        if dim_idx != x.ndim - 1:
            x = x.movedim(dim_idx, -1)

        # Apply biquad filter multiple times for higher order
        for _ in range(self.order):
            x = AF.lowpass_biquad(x, self.fs, self.cutoff, self.Q)

        # Move time dimension back
        if dim_idx != x.ndim - 1:
            x = x.movedim(-1, dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x


class Notch(TensorTransform):
    """Notch filter using FFT (GPU-accelerated).

    Removes a specific frequency (e.g., powerline interference at 50/60 Hz).
    Uses FFT-based approach which provides sharp, precise frequency removal
    ideal for narrow-band interference like powerline noise.

    Parameters
    ----------
    freq : float
        Center frequency to remove in Hz.
    width : float
        Width of the notch in Hz (default: 2 Hz).
    fs : float
        Sampling frequency in Hz.
    dim : str
        Dimension to filter over.
    """

    def __init__(
        self,
        freq: float,
        width: float = 2.0,
        fs: float = 2048.0,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.freq = freq
        self.width = width
        self.fs = fs

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)

        X = torch.fft.rfft(x, dim=dim_idx)
        freqs = torch.fft.rfftfreq(n_samples, 1 / self.fs, device=x.device)

        # Create notch (inverse of bandpass around freq)
        half_width = self.width / 2
        mask = ~((freqs >= self.freq - half_width) & (freqs <= self.freq + half_width))
        mask = mask.float()

        shape = [1] * x.ndim
        shape[dim_idx] = -1
        mask = mask.view(*shape)

        X_filtered = X * mask
        x_filtered = torch.fft.irfft(X_filtered, n=n_samples, dim=dim_idx)

        if names[0] is not None:
            x_filtered = x_filtered.rename(*names)

        return x_filtered


class ZeroCrossings(TensorTransform):
    """Count zero crossings in sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # Count sign changes
        signs = torch.sign(x_unfolded)
        zc = torch.sum(torch.abs(torch.diff(signs, dim=-1)) > 0, dim=-1).float()

        if names[0] is not None:
            zc = zc.rename(*names)

        return zc


class SlopeSignChanges(TensorTransform):
    """Count slope sign changes in sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # Compute differences (slope)
        slopes = torch.diff(x_unfolded, dim=-1)

        # Count sign changes in slopes
        signs = torch.sign(slopes)
        ssc = torch.sum(torch.abs(torch.diff(signs, dim=-1)) > 0, dim=-1).float()

        if names[0] is not None:
            ssc = ssc.rename(*names)

        return ssc


class WaveformLength(TensorTransform):
    """Waveform length over sliding windows (GPU-accelerated).

    Sum of absolute differences between consecutive samples.

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = 'time',
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        x_unfolded = x.unfold(dim_idx, self.window_size, self.stride)

        # Waveform length: sum(|x[n] - x[n-1]|)
        wfl = torch.sum(torch.abs(torch.diff(x_unfolded, dim=-1)), dim=-1)

        if names[0] is not None:
            wfl = wfl.rename(*names)

        return wfl


class Diff(TensorTransform):
    """Compute differences along a dimension.

    Parameters
    ----------
    n : int
        Number of times to differentiate.
    dim : str
        Dimension to differentiate over.
    """

    def __init__(self, n: int = 1, dim: str = 'time', **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.n = n

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        for _ in range(self.n):
            x = torch.diff(x, dim=dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x
