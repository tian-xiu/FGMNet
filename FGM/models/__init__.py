"""
Package init for models
"""
from .fgmnet import FGMNet
from .srm import SRM
from .fam import FAM
from .fai import FAI
from .cmfusion import CMFusion

__all__ = [
    'FGMNet',
    'SRM',
    'FAM',
    'FAI',
    'CMFusion',
]
