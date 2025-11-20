# src/core/__init__.py
"""Módulos core del proyecto."""

from .material_permutator import (
    MaterialPermutator,
    PermutationFilter,
    PermutationResult,
    MATERIAL_PERMUTATOR,
    generate_all_iii_v,
    generate_all_ii_vi,
    generate_all_semiconductors
)

from .multi_material_config import (
    MaterialConfig,
    MultiMaterialConfig,
    create_iii_v_config,
    create_ii_vi_config,
    create_common_semiconductors_config
)

__all__ = [
    # Material Permutator
    'MaterialPermutator',
    'PermutationFilter',
    'PermutationResult',
    'MATERIAL_PERMUTATOR',
    'generate_all_iii_v',
    'generate_all_ii_vi',
    'generate_all_semiconductors',
    
    # Multi-Material Config
    'MaterialConfig',
    'MultiMaterialConfig',
    'create_iii_v_config',
    'create_ii_vi_config',
    'create_common_semiconductors_config'
]
