"""
Stand-Alone Self-Attention in Vision Models

Reference: https://papers.nips.cc/paper/8302-stand-alone-self-attention-in-vision-models.pdf

Attention can be a stand-alone primitive for vision models instead of
serving as just an augmentation on top of convolutions
"""

from .model import ResNet26, ResNet38, ResNet50
from .attention import AttentionConv2d, AttentionStem

__all__ = [
    'ResNet26', 'ResNet38', 'ResNet50',
    'AttentionStem', 'AttentionConv2d',
]
