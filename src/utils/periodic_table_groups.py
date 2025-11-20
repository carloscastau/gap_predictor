# src/utils/periodic_table_groups.py
"""Base de datos de elementos de la tabla periódica para semiconductores III-V y II-VI."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PeriodicGroup(Enum):
    """Grupos de la tabla periódica relevantes para semiconductores."""
    GROUP_II = "II"
    GROUP_III = "III"
    GROUP_V = "V"
    GROUP_VI = "VI"


@dataclass
class ElementData:
    """Datos de un elemento químico."""
    symbol: str
    name: str
    atomic_number: int
    group: PeriodicGroup
    atomic_mass: float  # u (uma)
    ionic_radius: float  # Å (angstroms)
    electronegativity: float  # Escala de Pauling
    electron_config: str
    oxidation_states: List[int]
    covalent_radius: float  # Å
    melting_point: Optional[float] = None  # K
    common_in_semiconductors: bool = True
    
    def __str__(self) -> str:
        return f"{self.symbol} ({self.name})"
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'atomic_number': self.atomic_number,
            'group': self.group.value,
            'atomic_mass': self.atomic_mass,
            'ionic_radius': self.ionic_radius,
            'electronegativity': self.electronegativity,
            'electron_config': self.electron_config,
            'oxidation_states': self.oxidation_states,
            'covalent_radius': self.covalent_radius,
            'melting_point': self.melting_point,
            'common_in_semiconductors': self.common_in_semiconductors
        }


# Base de datos de elementos del Grupo III
GROUP_III_ELEMENTS: Dict[str, ElementData] = {
    'B': ElementData(
        symbol='B',
        name='Boro',
        atomic_number=5,
        group=PeriodicGroup.GROUP_III,
        atomic_mass=10.81,
        ionic_radius=0.27,  # B³⁺
        electronegativity=2.04,
        electron_config='[He] 2s² 2p¹',
        oxidation_states=[3],
        covalent_radius=0.84,
        melting_point=2349.0,
        common_in_semiconductors=True
    ),
    'Al': ElementData(
        symbol='Al',
        name='Aluminio',
        atomic_number=13,
        group=PeriodicGroup.GROUP_III,
        atomic_mass=26.98,
        ionic_radius=0.535,  # Al³⁺
        electronegativity=1.61,
        electron_config='[Ne] 3s² 3p¹',
        oxidation_states=[3],
        covalent_radius=1.21,
        melting_point=933.5,
        common_in_semiconductors=True
    ),
    'Ga': ElementData(
        symbol='Ga',
        name='Galio',
        atomic_number=31,
        group=PeriodicGroup.GROUP_III,
        atomic_mass=69.72,
        ionic_radius=0.62,  # Ga³⁺
        electronegativity=1.81,
        electron_config='[Ar] 3d¹⁰ 4s² 4p¹',
        oxidation_states=[3],
        covalent_radius=1.22,
        melting_point=302.9,
        common_in_semiconductors=True
    ),
    'In': ElementData(
        symbol='In',
        name='Indio',
        atomic_number=49,
        group=PeriodicGroup.GROUP_III,
        atomic_mass=114.82,
        ionic_radius=0.80,  # In³⁺
        electronegativity=1.78,
        electron_config='[Kr] 4d¹⁰ 5s² 5p¹',
        oxidation_states=[3],
        covalent_radius=1.42,
        melting_point=429.7,
        common_in_semiconductors=True
    ),
    'Tl': ElementData(
        symbol='Tl',
        name='Talio',
        atomic_number=81,
        group=PeriodicGroup.GROUP_III,
        atomic_mass=204.38,
        ionic_radius=0.885,  # Tl³⁺
        electronegativity=1.62,
        electron_config='[Xe] 4f¹⁴ 5d¹⁰ 6s² 6p¹',
        oxidation_states=[1, 3],
        covalent_radius=1.45,
        melting_point=577.0,
        common_in_semiconductors=False  # Menos común, tóxico
    )
}


# Base de datos de elementos del Grupo V
GROUP_V_ELEMENTS: Dict[str, ElementData] = {
    'N': ElementData(
        symbol='N',
        name='Nitrógeno',
        atomic_number=7,
        group=PeriodicGroup.GROUP_V,
        atomic_mass=14.01,
        ionic_radius=0.146,  # N³⁻
        electronegativity=3.04,
        electron_config='[He] 2s² 2p³',
        oxidation_states=[-3, 3, 5],
        covalent_radius=0.71,
        melting_point=63.15,
        common_in_semiconductors=True
    ),
    'P': ElementData(
        symbol='P',
        name='Fósforo',
        atomic_number=15,
        group=PeriodicGroup.GROUP_V,
        atomic_mass=30.97,
        ionic_radius=0.212,  # P³⁻
        electronegativity=2.19,
        electron_config='[Ne] 3s² 3p³',
        oxidation_states=[-3, 3, 5],
        covalent_radius=1.07,
        melting_point=317.3,
        common_in_semiconductors=True
    ),
    'As': ElementData(
        symbol='As',
        name='Arsénico',
        atomic_number=33,
        group=PeriodicGroup.GROUP_V,
        atomic_mass=74.92,
        ionic_radius=0.58,  # As³⁻
        electronegativity=2.18,
        electron_config='[Ar] 3d¹⁰ 4s² 4p³',
        oxidation_states=[-3, 3, 5],
        covalent_radius=1.19,
        melting_point=1090.0,
        common_in_semiconductors=True
    ),
    'Sb': ElementData(
        symbol='Sb',
        name='Antimonio',
        atomic_number=51,
        group=PeriodicGroup.GROUP_V,
        atomic_mass=121.76,
        ionic_radius=0.76,  # Sb³⁻
        electronegativity=2.05,
        electron_config='[Kr] 4d¹⁰ 5s² 5p³',
        oxidation_states=[-3, 3, 5],
        covalent_radius=1.39,
        melting_point=903.8,
        common_in_semiconductors=True
    ),
    'Bi': ElementData(
        symbol='Bi',
        name='Bismuto',
        atomic_number=83,
        group=PeriodicGroup.GROUP_V,
        atomic_mass=208.98,
        ionic_radius=1.03,  # Bi³⁻
        electronegativity=2.02,
        electron_config='[Xe] 4f¹⁴ 5d¹⁰ 6s² 6p³',
        oxidation_states=[-3, 3, 5],
        covalent_radius=1.48,
        melting_point=544.4,
        common_in_semiconductors=True
    )
}


# Base de datos de elementos del Grupo II
GROUP_II_ELEMENTS: Dict[str, ElementData] = {
    'Be': ElementData(
        symbol='Be',
        name='Berilio',
        atomic_number=4,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=9.01,
        ionic_radius=0.45,  # Be²⁺
        electronegativity=1.57,
        electron_config='[He] 2s²',
        oxidation_states=[2],
        covalent_radius=0.96,
        melting_point=1560.0,
        common_in_semiconductors=False  # Tóxico
    ),
    'Mg': ElementData(
        symbol='Mg',
        name='Magnesio',
        atomic_number=12,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=24.31,
        ionic_radius=0.72,  # Mg²⁺
        electronegativity=1.31,
        electron_config='[Ne] 3s²',
        oxidation_states=[2],
        covalent_radius=1.41,
        melting_point=923.0,
        common_in_semiconductors=True
    ),
    'Ca': ElementData(
        symbol='Ca',
        name='Calcio',
        atomic_number=20,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=40.08,
        ionic_radius=1.00,  # Ca²⁺
        electronegativity=1.00,
        electron_config='[Ar] 4s²',
        oxidation_states=[2],
        covalent_radius=1.76,
        melting_point=1115.0,
        common_in_semiconductors=False
    ),
    'Sr': ElementData(
        symbol='Sr',
        name='Estroncio',
        atomic_number=38,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=87.62,
        ionic_radius=1.18,  # Sr²⁺
        electronegativity=0.95,
        electron_config='[Kr] 5s²',
        oxidation_states=[2],
        covalent_radius=1.95,
        melting_point=1050.0,
        common_in_semiconductors=False
    ),
    'Ba': ElementData(
        symbol='Ba',
        name='Bario',
        atomic_number=56,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=137.33,
        ionic_radius=1.35,  # Ba²⁺
        electronegativity=0.89,
        electron_config='[Xe] 6s²',
        oxidation_states=[2],
        covalent_radius=2.15,
        melting_point=1000.0,
        common_in_semiconductors=False
    ),
    'Zn': ElementData(
        symbol='Zn',
        name='Zinc',
        atomic_number=30,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=65.38,
        ionic_radius=0.74,  # Zn²⁺
        electronegativity=1.65,
        electron_config='[Ar] 3d¹⁰ 4s²',
        oxidation_states=[2],
        covalent_radius=1.22,
        melting_point=692.7,
        common_in_semiconductors=True
    ),
    'Cd': ElementData(
        symbol='Cd',
        name='Cadmio',
        atomic_number=48,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=112.41,
        ionic_radius=0.95,  # Cd²⁺
        electronegativity=1.69,
        electron_config='[Kr] 4d¹⁰ 5s²',
        oxidation_states=[2],
        covalent_radius=1.44,
        melting_point=594.2,
        common_in_semiconductors=True
    ),
    'Hg': ElementData(
        symbol='Hg',
        name='Mercurio',
        atomic_number=80,
        group=PeriodicGroup.GROUP_II,
        atomic_mass=200.59,
        ionic_radius=1.02,  # Hg²⁺
        electronegativity=2.00,
        electron_config='[Xe] 4f¹⁴ 5d¹⁰ 6s²',
        oxidation_states=[1, 2],
        covalent_radius=1.32,
        melting_point=234.3,
        common_in_semiconductors=True
    )
}


# Base de datos de elementos del Grupo VI
GROUP_VI_ELEMENTS: Dict[str, ElementData] = {
    'O': ElementData(
        symbol='O',
        name='Oxígeno',
        atomic_number=8,
        group=PeriodicGroup.GROUP_VI,
        atomic_mass=16.00,
        ionic_radius=1.40,  # O²⁻
        electronegativity=3.44,
        electron_config='[He] 2s² 2p⁴',
        oxidation_states=[-2],
        covalent_radius=0.66,
        melting_point=54.8,
        common_in_semiconductors=True
    ),
    'S': ElementData(
        symbol='S',
        name='Azufre',
        atomic_number=16,
        group=PeriodicGroup.GROUP_VI,
        atomic_mass=32.07,
        ionic_radius=1.84,  # S²⁻
        electronegativity=2.58,
        electron_config='[Ne] 3s² 3p⁴',
        oxidation_states=[-2, 4, 6],
        covalent_radius=1.05,
        melting_point=388.4,
        common_in_semiconductors=True
    ),
    'Se': ElementData(
        symbol='Se',
        name='Selenio',
        atomic_number=34,
        group=PeriodicGroup.GROUP_VI,
        atomic_mass=78.96,
        ionic_radius=1.98,  # Se²⁻
        electronegativity=2.55,
        electron_config='[Ar] 3d¹⁰ 4s² 4p⁴',
        oxidation_states=[-2, 4, 6],
        covalent_radius=1.20,
        melting_point=494.0,
        common_in_semiconductors=True
    ),
    'Te': ElementData(
        symbol='Te',
        name='Telurio',
        atomic_number=52,
        group=PeriodicGroup.GROUP_VI,
        atomic_mass=127.60,
        ionic_radius=2.21,  # Te²⁻
        electronegativity=2.10,
        electron_config='[Kr] 4d¹⁰ 5s² 5p⁴',
        oxidation_states=[-2, 4, 6],
        covalent_radius=1.38,
        melting_point=722.7,
        common_in_semiconductors=True
    ),
    'Po': ElementData(
        symbol='Po',
        name='Polonio',
        atomic_number=84,
        group=PeriodicGroup.GROUP_VI,
        atomic_mass=209.0,
        ionic_radius=2.30,  # Po²⁻ (estimado)
        electronegativity=2.0,
        electron_config='[Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁴',
        oxidation_states=[-2, 2, 4],
        covalent_radius=1.40,
        melting_point=527.0,
        common_in_semiconductors=False  # Radiactivo
    )
}


class PeriodicTableDatabase:
    """Base de datos completa de elementos para semiconductores."""
    
    def __init__(self):
        """Inicializa la base de datos."""
        self.group_ii = GROUP_II_ELEMENTS
        self.group_iii = GROUP_III_ELEMENTS
        self.group_v = GROUP_V_ELEMENTS
        self.group_vi = GROUP_VI_ELEMENTS
        
    def get_element(self, symbol: str) -> Optional[ElementData]:
        """Obtiene datos de un elemento por su símbolo."""
        all_elements = {
            **self.group_ii,
            **self.group_iii,
            **self.group_v,
            **self.group_vi
        }
        return all_elements.get(symbol)
    
    def get_group_elements(self, group: PeriodicGroup) -> Dict[str, ElementData]:
        """Obtiene todos los elementos de un grupo."""
        group_map = {
            PeriodicGroup.GROUP_II: self.group_ii,
            PeriodicGroup.GROUP_III: self.group_iii,
            PeriodicGroup.GROUP_V: self.group_v,
            PeriodicGroup.GROUP_VI: self.group_vi
        }
        return group_map.get(group, {})
    
    def get_common_elements(self, group: PeriodicGroup) -> Dict[str, ElementData]:
        """Obtiene elementos comunes en semiconductores de un grupo."""
        elements = self.get_group_elements(group)
        return {
            symbol: data for symbol, data in elements.items()
            if data.common_in_semiconductors
        }
    
    def get_all_elements(self) -> Dict[str, ElementData]:
        """Obtiene todos los elementos de la base de datos."""
        return {
            **self.group_ii,
            **self.group_iii,
            **self.group_v,
            **self.group_vi
        }
    
    def get_elements_by_atomic_number(self, min_z: int, max_z: int) -> List[ElementData]:
        """Obtiene elementos en un rango de números atómicos."""
        all_elements = self.get_all_elements()
        return [
            data for data in all_elements.values()
            if min_z <= data.atomic_number <= max_z
        ]
    
    def __repr__(self) -> str:
        return (f"PeriodicTableDatabase("
                f"II: {len(self.group_ii)}, "
                f"III: {len(self.group_iii)}, "
                f"V: {len(self.group_v)}, "
                f"VI: {len(self.group_vi)})")


# Instancia global de la base de datos
PERIODIC_TABLE = PeriodicTableDatabase()


# Funciones de utilidad
def get_element(symbol: str) -> Optional[ElementData]:
    """Función de conveniencia para obtener un elemento."""
    return PERIODIC_TABLE.get_element(symbol)


def get_group_iii_elements() -> Dict[str, ElementData]:
    """Obtiene elementos del grupo III."""
    return PERIODIC_TABLE.group_iii


def get_group_v_elements() -> Dict[str, ElementData]:
    """Obtiene elementos del grupo V."""
    return PERIODIC_TABLE.group_v


def get_group_ii_elements() -> Dict[str, ElementData]:
    """Obtiene elementos del grupo II."""
    return PERIODIC_TABLE.group_ii


def get_group_vi_elements() -> Dict[str, ElementData]:
    """Obtiene elementos del grupo VI."""
    return PERIODIC_TABLE.group_vi
