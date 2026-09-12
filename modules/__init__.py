"""
iNAV Python Modules Package
Core mathematical estimators, neural networks, alignment routines, and map matching.
"""

from .alignment import AlignmentEngine
from .velocity_net import VelocityNetPredictor
from .gnss_handler import GnssHandler, NavigationMode
from .ukf import UKFNavigationFilter
from .esekf import ESEKFNavigationFilter, ESEKFNavigationFilter as ESEKF
from .map_matcher import HMMMapMatcher, SpatialGridIndex, RoadSegment
try:
    from .mtn import MotionTransformationNetwork
except ImportError:
    MotionTransformationNetwork = None

from .sensor_types import (
    SensorFrame,
    Vec3,
    ImuSample,
    GnssSample,
    ImuValidity,
    GnssValidity,
    units,
    make_android_imu_sample,
)
from .dataset_adapter import (
    row_to_canonical_imu,
    row_to_canonical_gnss,
    dataframe_to_canonical_samples,
    iter_canonical_samples,
)

__all__ = [
    "AlignmentEngine",
    "VelocityNetPredictor",
    "GnssHandler",
    "NavigationMode",
    "UKFNavigationFilter",
    "ESEKFNavigationFilter",
    "ESEKF",
    "HMMMapMatcher",
    "SpatialGridIndex",
    "RoadSegment",
    "MotionTransformationNetwork",
    "SensorFrame",
    "Vec3",
    "ImuSample",
    "GnssSample",
    "ImuValidity",
    "GnssValidity",
    "units",
    "make_android_imu_sample",
    "row_to_canonical_imu",
    "row_to_canonical_gnss",
    "dataframe_to_canonical_samples",
    "iter_canonical_samples",
]
