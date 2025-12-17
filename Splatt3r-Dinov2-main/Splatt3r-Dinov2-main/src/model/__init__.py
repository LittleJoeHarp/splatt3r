"""
Splatt3R Model Package

This package contains the main Splatt3R architecture with DINOv2 support.
"""

from .splatt3r import Splatt3R, Splatt3RFactory
from .feature_fusion import (
    DINOv2FeatureFusionAdapter,
    DINOv2DecoderFeatureFusion,
    DINOv2ConfigurableInputHead,
)
from .dinov2_integration_guide import (
    DINOv2IntegrationGuide,
    MASt3RHeadDimensionValidator,
    CONFIG_TEMPLATES,
    print_configuration_templates,
)

__all__ = [
    'Splatt3R',
    'Splatt3RFactory',
    'DINOv2FeatureFusionAdapter',
    'DINOv2DecoderFeatureFusion',
    'DINOv2ConfigurableInputHead',
    'DINOv2IntegrationGuide',
    'MASt3RHeadDimensionValidator',
    'CONFIG_TEMPLATES',
    'print_configuration_templates',
]
