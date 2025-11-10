# src/models/basis.py
"""Modelos de datos para bases y pseudopotenciales."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import os


@dataclass
class BasisSetModel:
    """Modelo de conjunto de base."""

    name: str
    family: str = ""  # gth, def2, etc.
    quality: str = ""  # sz, dz, tz, etc.
    elements: Set[str] = field(default_factory=set)
    n_functions: Dict[str, int] = field(default_factory=dict)
    is_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Inicializa propiedades derivadas."""
        if not self.family and self.name:
            self._infer_family_and_quality()

    def _infer_family_and_quality(self):
        """Infiera familia y calidad desde el nombre."""
        name_lower = self.name.lower()

        # Inferir familia
        if 'gth' in name_lower:
            self.family = 'gth'
        elif 'def2' in name_lower:
            self.family = 'def2'
        elif 'cc-pv' in name_lower:
            self.family = 'cc-pv'
        elif 'aug-cc-pv' in name_lower:
            self.family = 'aug-cc-pv'

        # Inferir calidad
        if 'sz' in name_lower or 'sv' in name_lower:
            self.quality = 'sz'
        elif 'dz' in name_lower:
            self.quality = 'dz'
        elif 'tz' in name_lower:
            self.quality = 'tz'
        elif 'qz' in name_lower:
            self.quality = 'qz'

    @property
    def is_gth_family(self) -> bool:
        """Verifica si es familia GTH."""
        return self.family == 'gth'

    @property
    def quality_order(self) -> int:
        """Orden de calidad (menor = más pequeño)."""
        quality_map = {'sz': 1, 'dz': 2, 'tz': 3, 'qz': 4}
        return quality_map.get(self.quality, 0)

    def supports_element(self, element: str) -> bool:
        """Verifica si la base soporta un elemento."""
        return element in self.elements

    def get_n_functions_for_element(self, element: str) -> int:
        """Obtiene número de funciones para un elemento."""
        return self.n_functions.get(element, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'name': self.name,
            'family': self.family,
            'quality': self.quality,
            'elements': list(self.elements),
            'n_functions': self.n_functions,
            'is_available': self.is_available,
            'metadata': self.metadata
        }


@dataclass
class PseudopotentialModel:
    """Modelo de pseudopotencial."""

    name: str
    family: str = ""  # gth, ultrasoft, paw, etc.
    elements: Set[str] = field(default_factory=set)
    is_norm_conserving: bool = True
    is_ultrasoft: bool = False
    is_paw: bool = False
    valence_electrons: Dict[str, int] = field(default_factory=dict)
    is_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Inicializa propiedades derivadas."""
        if not self.family and self.name:
            self._infer_family()

    def _infer_family(self):
        """Infiera familia desde el nombre."""
        name_lower = self.name.lower()

        if 'gth' in name_lower:
            self.family = 'gth'
        elif 'uspp' in name_lower or 'ultrasoft' in name_lower:
            self.family = 'ultrasoft'
            self.is_ultrasoft = True
            self.is_norm_conserving = False
        elif 'paw' in name_lower:
            self.family = 'paw'
            self.is_paw = True
            self.is_norm_conserving = False

    @property
    def is_gth_family(self) -> bool:
        """Verifica si es familia GTH."""
        return self.family == 'gth'

    def supports_element(self, element: str) -> bool:
        """Verifica si el pseudopotencial soporta un elemento."""
        return element in self.elements

    def get_valence_electrons(self, element: str) -> int:
        """Obtiene electrones de valencia para un elemento."""
        return self.valence_electrons.get(element, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'name': self.name,
            'family': self.family,
            'elements': list(self.elements),
            'is_norm_conserving': self.is_norm_conserving,
            'is_ultrasoft': self.is_ultrasoft,
            'is_paw': self.is_paw,
            'valence_electrons': self.valence_electrons,
            'is_available': self.is_available,
            'metadata': self.metadata
        }


# Bases GTH disponibles en PySCF
GTH_BASES = {
    'gth-szv': BasisSetModel(
        name='gth-szv',
        family='gth',
        quality='sz',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        n_functions={'Ga': 13, 'As': 13}  # Estimaciones
    ),
    'gth-dzv': BasisSetModel(
        name='gth-dzv',
        family='gth',
        quality='dz',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        n_functions={'Ga': 18, 'As': 18}
    ),
    'gth-dzvp': BasisSetModel(
        name='gth-dzvp',
        family='gth',
        quality='dz',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        n_functions={'Ga': 23, 'As': 23}  # Incluye funciones de polarización
    ),
    'gth-tzvp': BasisSetModel(
        name='gth-tzvp',
        family='gth',
        quality='tz',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        n_functions={'Ga': 30, 'As': 30}
    )
}

# Pseudopotenciales GTH disponibles
GTH_PSEUDOS = {
    'gth-pbe': PseudopotentialModel(
        name='gth-pbe',
        family='gth',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        valence_electrons={'Ga': 13, 'As': 5}  # Ga: [Ar] 3d10 4s2 4p1, As: [Ar] 3d10 4s2 4p3
    ),
    'gth-pbesol': PseudopotentialModel(
        name='gth-pbesol',
        family='gth',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        valence_electrons={'Ga': 13, 'As': 5}
    ),
    'gth-lda': PseudopotentialModel(
        name='gth-lda',
        family='gth',
        elements={'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'},
        valence_electrons={'Ga': 13, 'As': 5}
    )
}


def get_available_gth_bases() -> List[BasisSetModel]:
    """Obtiene lista de bases GTH disponibles."""
    return list(GTH_BASES.values())


def get_available_gth_pseudos() -> List[PseudopotentialModel]:
    """Obtiene lista de pseudopotenciales GTH disponibles."""
    return list(GTH_PSEUDOS.values())


def get_basis_by_name(name: str) -> Optional[BasisSetModel]:
    """Obtiene base por nombre."""
    return GTH_BASES.get(name.lower())


def get_pseudo_by_name(name: str) -> Optional[PseudopotentialModel]:
    """Obtiene pseudopotencial por nombre."""
    return GTH_PSEUDOS.get(name.lower())


def validate_basis_pseudo_compatibility(basis: BasisSetModel,
                                       pseudo: PseudopotentialModel) -> bool:
    """Valida compatibilidad entre base y pseudopotencial."""
    # Para GTH, base y pseudo deben ser de la misma familia
    if basis.is_gth_family and pseudo.is_gth_family:
        return True

    # Para otras familias, asumir compatibilidad por ahora
    return True


def get_recommended_basis_for_accuracy(target_accuracy: str = 'normal') -> BasisSetModel:
    """Obtiene base recomendada para nivel de precisión dado."""
    accuracy_map = {
        'low': 'gth-dzv',
        'normal': 'gth-dzvp',
        'high': 'gth-tzvp',
        'very_high': 'gth-tzv2p'
    }

    basis_name = accuracy_map.get(target_accuracy, 'gth-dzvp')
    return get_basis_by_name(basis_name) or GTH_BASES['gth-dzvp']