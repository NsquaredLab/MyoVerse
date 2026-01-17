"""RaulNet model family for EMG-to-kinematics decoding.

This module contains the RaulNet model architectures used for decoding
hand kinematics from high-density EMG signals.

References
----------
.. [1] Sîmpetru et al. (2024) MyoGestic: EMG interfacing framework for decoding
       multiple spared degrees of freedom of the hand in individuals with neural lesions.

"""

from myoverse.models.raul_net.v16 import RaulNetV16
from myoverse.models.raul_net.v17 import RaulNetV17

__all__ = ["RaulNetV16", "RaulNetV17"]
