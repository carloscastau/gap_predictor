# src/models/semiconductor_database.py
"""Base de datos de semiconductores binarios III-V y II-VI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from ..utils.periodic_table_groups import (
    ElementData,
    PeriodicGroup,
    PERIODIC_TABLE,
    get_element
)


class SemiconductorType(Enum):
    """Tipos de semiconductores binarios."""
    III_V = "III-V"
    II_VI = "II-VI"
    UNKNOWN = "Unknown"


class CrystalStructure(Enum):
    """Estructuras cristalinas comunes."""
    ZINCBLENDE = "Zincblende"  # Cúbica (sphalerite)
    WURTZITE = "Wurtzite"      # Hexagonal
    ROCKSALT = "Rocksalt"      # Cúbica (NaCl)
    UNKNOWN = "Unknown"


@dataclass
class SemiconductorProperties:
    """Propiedades físicas de un semiconductor binario."""
    formula: str
    lattice_constant: Optional[float] = None  # Å
    band_gap: Optional[float] = None  # eV
    crystal_structure: CrystalStructure = CrystalStructure.ZINCBLENDE
    density: Optional[float] = None  # g/cm³
    melting_point: Optional[float] = None  # K
    
    # Propiedades electrónicas
    electron_mobility: Optional[float] = None  # cm²/V·s
    hole_mobility: Optional[float] = None  # cm²/V·s
    dielectric_constant: Optional[float] = None
    
    # Propiedades térmicas
    thermal_conductivity: Optional[float] = None  # W/m·K
    thermal_expansion: Optional[float] = None  # 10⁻⁶/K
    
    # Metadatos
    is_stable: bool = True
    is_experimental: bool = False
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'formula': self.formula,
            'lattice_constant': self.lattice_constant,
            'band_gap': self.band_gap,
            'crystal_structure': self.crystal_structure.value,
            'density': self.density,
            'melting_point': self.melting_point,
            'electron_mobility': self.electron_mobility,
            'hole_mobility': self.hole_mobility,
            'dielectric_constant': self.dielectric_constant,
            'thermal_conductivity': self.thermal_conductivity,
            'thermal_expansion': self.thermal_expansion,
            'is_stable': self.is_stable,
            'is_experimental': self.is_experimental,
            'notes': self.notes
        }


@dataclass
class BinarySemiconductor:
    """Representa un semiconductor binario."""
    cation: ElementData
    anion: ElementData
    semiconductor_type: SemiconductorType
    properties: Optional[SemiconductorProperties] = None
    
    def __post_init__(self):
        """Validación y configuración inicial."""
        self._validate_composition()
        if self.properties is None:
            self.properties = SemiconductorProperties(
                formula=self.formula,
                is_experimental=True
            )
    
    def _validate_composition(self):
        """Valida que la composición sea correcta."""
        if self.semiconductor_type == SemiconductorType.III_V:
            if self.cation.group != PeriodicGroup.GROUP_III:
                raise ValueError(f"Catión {self.cation.symbol} no es del grupo III")
            if self.anion.group != PeriodicGroup.GROUP_V:
                raise ValueError(f"Anión {self.anion.symbol} no es del grupo V")
        elif self.semiconductor_type == SemiconductorType.II_VI:
            if self.cation.group != PeriodicGroup.GROUP_II:
                raise ValueError(f"Catión {self.cation.symbol} no es del grupo II")
            if self.anion.group != PeriodicGroup.GROUP_VI:
                raise ValueError(f"Anión {self.anion.symbol} no es del grupo VI")
    
    @property
    def formula(self) -> str:
        """Fórmula química del compuesto."""
        return f"{self.cation.symbol}{self.anion.symbol}"
    
    @property
    def ionic_radius_ratio(self) -> float:
        """Razón de radios iónicos (catión/anión)."""
        return self.cation.ionic_radius / self.anion.ionic_radius
    
    @property
    def electronegativity_difference(self) -> float:
        """Diferencia de electronegatividad."""
        return abs(self.anion.electronegativity - self.cation.electronegativity)
    
    @property
    def average_atomic_mass(self) -> float:
        """Masa atómica promedio."""
        return (self.cation.atomic_mass + self.anion.atomic_mass) / 2
    
    @property
    def predicted_crystal_structure(self) -> CrystalStructure:
        """Predice estructura cristalina basada en radio iónico."""
        ratio = self.ionic_radius_ratio
        
        # Reglas de Pauling simplificadas
        if 0.225 <= ratio <= 0.414:
            return CrystalStructure.ZINCBLENDE
        elif 0.414 <= ratio <= 0.732:
            return CrystalStructure.ZINCBLENDE  # o Wurtzite
        elif ratio > 0.732:
            return CrystalStructure.ROCKSALT
        else:
            return CrystalStructure.UNKNOWN
    
    def estimate_lattice_constant(self) -> float:
        """Estima constante de red usando radios covalentes."""
        # Aproximación: a ≈ 2√2 * (r_cation + r_anion) para zincblende
        r_sum = self.cation.covalent_radius + self.anion.covalent_radius
        return 2.0 * np.sqrt(2.0) * r_sum
    
    def is_chemically_compatible(self, 
                                 max_radius_ratio: float = 2.0,
                                 min_electronegativity_diff: float = 0.3) -> bool:
        """Verifica compatibilidad química básica."""
        # Verificar razón de radios
        if self.ionic_radius_ratio > max_radius_ratio:
            return False
        
        # Verificar diferencia de electronegatividad mínima
        if self.electronegativity_difference < min_electronegativity_diff:
            return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'formula': self.formula,
            'cation': self.cation.symbol,
            'anion': self.anion.symbol,
            'type': self.semiconductor_type.value,
            'ionic_radius_ratio': self.ionic_radius_ratio,
            'electronegativity_difference': self.electronegativity_difference,
            'average_atomic_mass': self.average_atomic_mass,
            'predicted_structure': self.predicted_crystal_structure.value,
            'estimated_lattice_constant': self.estimate_lattice_constant(),
            'is_compatible': self.is_chemically_compatible(),
            'properties': self.properties.to_dict() if self.properties else None
        }
    
    def __str__(self) -> str:
        return f"{self.formula} ({self.semiconductor_type.value})"
    
    def __repr__(self) -> str:
        return f"BinarySemiconductor({self.formula}, {self.semiconductor_type.value})"


class SemiconductorDatabase:
    """Base de datos completa de semiconductores binarios."""
    
    def __init__(self):
        """Inicializa la base de datos."""
        self.semiconductors: Dict[str, BinarySemiconductor] = {}
        self._initialize_known_semiconductors()
    
    def _initialize_known_semiconductors(self):
        """Inicializa semiconductores conocidos con propiedades experimentales."""
        
        # Semiconductores III-V conocidos
        known_iii_v = {
            'GaAs': SemiconductorProperties(
                formula='GaAs',
                lattice_constant=5.653,
                band_gap=1.424,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.316,
                melting_point=1511.0,
                electron_mobility=8500.0,
                hole_mobility=400.0,
                dielectric_constant=12.9,
                thermal_conductivity=55.0,
                thermal_expansion=5.73,
                is_stable=True,
                is_experimental=False,
                notes="Semiconductor III-V más estudiado"
            ),
            'GaN': SemiconductorProperties(
                formula='GaN',
                lattice_constant=4.52,  # wurtzite a-axis
                band_gap=3.4,
                crystal_structure=CrystalStructure.WURTZITE,
                density=6.15,
                melting_point=2791.0,
                electron_mobility=1000.0,
                hole_mobility=30.0,
                dielectric_constant=9.5,
                thermal_conductivity=130.0,
                is_stable=True,
                is_experimental=False,
                notes="Wide bandgap, LEDs y electrónica de potencia"
            ),
            'InP': SemiconductorProperties(
                formula='InP',
                lattice_constant=5.869,
                band_gap=1.344,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=4.81,
                melting_point=1335.0,
                electron_mobility=4600.0,
                hole_mobility=150.0,
                dielectric_constant=12.5,
                thermal_conductivity=68.0,
                is_stable=True,
                is_experimental=False,
                notes="Telecomunicaciones y fotónica"
            ),
            'AlAs': SemiconductorProperties(
                formula='AlAs',
                lattice_constant=5.661,
                band_gap=2.168,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=3.76,
                melting_point=2013.0,
                electron_mobility=280.0,
                dielectric_constant=10.1,
                is_stable=True,
                is_experimental=False,
                notes="Usado en heteroestructuras"
            ),
            'InAs': SemiconductorProperties(
                formula='InAs',
                lattice_constant=6.058,
                band_gap=0.354,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.667,
                melting_point=1215.0,
                electron_mobility=33000.0,
                hole_mobility=460.0,
                dielectric_constant=15.15,
                is_stable=True,
                is_experimental=False,
                notes="Alta movilidad electrónica"
            ),
            'GaP': SemiconductorProperties(
                formula='GaP',
                lattice_constant=5.451,
                band_gap=2.26,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=4.138,
                melting_point=1730.0,
                electron_mobility=250.0,
                hole_mobility=150.0,
                dielectric_constant=11.1,
                is_stable=True,
                is_experimental=False,
                notes="LEDs verdes y rojos"
            ),
            'InSb': SemiconductorProperties(
                formula='InSb',
                lattice_constant=6.479,
                band_gap=0.17,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.775,
                melting_point=798.0,
                electron_mobility=77000.0,
                hole_mobility=850.0,
                dielectric_constant=17.7,
                is_stable=True,
                is_experimental=False,
                notes="Mayor movilidad electrónica conocida"
            ),
            'AlN': SemiconductorProperties(
                formula='AlN',
                lattice_constant=4.38,
                band_gap=6.2,
                crystal_structure=CrystalStructure.WURTZITE,
                density=3.26,
                melting_point=3273.0,
                electron_mobility=300.0,
                dielectric_constant=8.5,
                thermal_conductivity=285.0,
                is_stable=True,
                is_experimental=False,
                notes="Wide bandgap, UV optoelectronics"
            ),
            'InN': SemiconductorProperties(
                formula='InN',
                lattice_constant=4.98,
                band_gap=0.7,
                crystal_structure=CrystalStructure.WURTZITE,
                density=6.81,
                melting_point=2146.0,
                electron_mobility=3200.0,
                dielectric_constant=15.3,
                is_stable=True,
                is_experimental=False,
                notes="Narrow bandgap III-nitride"
            ),
            'AlP': SemiconductorProperties(
                formula='AlP',
                lattice_constant=5.467,
                band_gap=2.45,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=2.40,
                melting_point=2823.0,
                electron_mobility=80.0,
                dielectric_constant=9.8,
                is_stable=True,
                is_experimental=False,
                notes="Indirect bandgap semiconductor"
            ),
            'GaSb': SemiconductorProperties(
                formula='GaSb',
                lattice_constant=6.096,
                band_gap=0.726,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.614,
                melting_point=985.0,
                electron_mobility=5000.0,
                hole_mobility=850.0,
                dielectric_constant=15.7,
                is_stable=True,
                is_experimental=False,
                notes="Infrared detectors and thermophotovoltaics"
            )
        }
        
        # Semiconductores II-VI conocidos
        known_ii_vi = {
            'ZnS': SemiconductorProperties(
                formula='ZnS',
                lattice_constant=5.409,
                band_gap=3.68,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=4.09,
                melting_point=2103.0,
                electron_mobility=165.0,
                dielectric_constant=8.9,
                is_stable=True,
                is_experimental=False,
                notes="Fosforescencia y pantallas"
            ),
            'ZnSe': SemiconductorProperties(
                formula='ZnSe',
                lattice_constant=5.668,
                band_gap=2.70,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.27,
                melting_point=1793.0,
                electron_mobility=530.0,
                hole_mobility=28.0,
                dielectric_constant=9.1,
                is_stable=True,
                is_experimental=False,
                notes="LEDs azul-verde"
            ),
            'ZnTe': SemiconductorProperties(
                formula='ZnTe',
                lattice_constant=6.104,
                band_gap=2.26,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.636,
                melting_point=1568.0,
                electron_mobility=340.0,
                hole_mobility=100.0,
                dielectric_constant=10.1,
                is_stable=True,
                is_experimental=False,
                notes="Detectores de radiación"
            ),
            'CdS': SemiconductorProperties(
                formula='CdS',
                lattice_constant=5.832,
                band_gap=2.42,
                crystal_structure=CrystalStructure.WURTZITE,
                density=4.826,
                melting_point=1748.0,
                electron_mobility=340.0,
                dielectric_constant=8.9,
                is_stable=True,
                is_experimental=False,
                notes="Fotoconductores y celdas solares"
            ),
            'CdSe': SemiconductorProperties(
                formula='CdSe',
                lattice_constant=6.05,
                band_gap=1.70,
                crystal_structure=CrystalStructure.WURTZITE,
                density=5.816,
                melting_point=1531.0,
                electron_mobility=800.0,
                dielectric_constant=10.2,
                is_stable=True,
                is_experimental=False,
                notes="Quantum dots y fotovoltaicos"
            ),
            'CdTe': SemiconductorProperties(
                formula='CdTe',
                lattice_constant=6.482,
                band_gap=1.50,
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=5.85,
                melting_point=1365.0,
                electron_mobility=1050.0,
                hole_mobility=100.0,
                dielectric_constant=10.9,
                is_stable=True,
                is_experimental=False,
                notes="Celdas solares de película delgada"
            ),
            'HgTe': SemiconductorProperties(
                formula='HgTe',
                lattice_constant=6.460,
                band_gap=0.0,  # Semimetal
                crystal_structure=CrystalStructure.ZINCBLENDE,
                density=8.17,
                melting_point=943.0,
                is_stable=True,
                is_experimental=False,
                notes="Semimetal, detectores IR"
            )
        }
        
        # Agregar semiconductores conocidos a la base de datos
        for formula, props in known_iii_v.items():
            # Parser mejorado para fórmulas químicas
            cation_symbol, anion_symbol = self._parse_formula(formula)
            
            cation = get_element(cation_symbol)
            anion = get_element(anion_symbol)
            
            if cation and anion:
                semiconductor = BinarySemiconductor(
                    cation=cation,
                    anion=anion,
                    semiconductor_type=SemiconductorType.III_V,
                    properties=props
                )
                self.semiconductors[formula] = semiconductor
        
        for formula, props in known_ii_vi.items():
            # Parser mejorado para fórmulas químicas
            cation_symbol, anion_symbol = self._parse_formula(formula)
            
            cation = get_element(cation_symbol)
            anion = get_element(anion_symbol)
            
            if cation and anion:
                semiconductor = BinarySemiconductor(
                    cation=cation,
                    anion=anion,
                    semiconductor_type=SemiconductorType.II_VI,
                    properties=props
                )
                self.semiconductors[formula] = semiconductor
    
    def _parse_formula(self, formula: str) -> tuple:
        """
        Parsea una fórmula química binaria.
        
        Args:
            formula: Fórmula química (ej: 'GaAs', 'ZnSe', 'AlN')
            
        Returns:
            Tupla (catión, anión)
        """
        import re
        # Buscar patrón: una o dos letras mayúsculas seguidas de minúsculas opcionales
        matches = re.findall(r'[A-Z][a-z]?', formula)
        if len(matches) == 2:
            return matches[0], matches[1]
        else:
            # Fallback para casos especiales
            if len(formula) == 4:
                return formula[:2], formula[2:]
            elif len(formula) == 3:
                return formula[0], formula[1:]
            else:
                return formula[0], formula[1]
    
    def add_semiconductor(self, semiconductor: BinarySemiconductor):
        """Agrega un semiconductor a la base de datos."""
        self.semiconductors[semiconductor.formula] = semiconductor
    
    def get_semiconductor(self, formula: str) -> Optional[BinarySemiconductor]:
        """Obtiene un semiconductor por su fórmula."""
        return self.semiconductors.get(formula)
    
    def get_all_iii_v(self) -> List[BinarySemiconductor]:
        """Obtiene todos los semiconductores III-V."""
        return [
            sc for sc in self.semiconductors.values()
            if sc.semiconductor_type == SemiconductorType.III_V
        ]
    
    def get_all_ii_vi(self) -> List[BinarySemiconductor]:
        """Obtiene todos los semiconductores II-VI."""
        return [
            sc for sc in self.semiconductors.values()
            if sc.semiconductor_type == SemiconductorType.II_VI
        ]
    
    def get_stable_semiconductors(self) -> List[BinarySemiconductor]:
        """Obtiene semiconductores estables conocidos."""
        return [
            sc for sc in self.semiconductors.values()
            if sc.properties and sc.properties.is_stable and not sc.properties.is_experimental
        ]
    
    def search_by_band_gap(self, min_gap: float, max_gap: float) -> List[BinarySemiconductor]:
        """Busca semiconductores por rango de band gap."""
        results = []
        for sc in self.semiconductors.values():
            if sc.properties and sc.properties.band_gap is not None:
                if min_gap <= sc.properties.band_gap <= max_gap:
                    results.append(sc)
        return results
    
    def search_by_lattice_constant(self, target: float, tolerance: float = 0.1) -> List[BinarySemiconductor]:
        """Busca semiconductores por constante de red."""
        results = []
        for sc in self.semiconductors.values():
            if sc.properties and sc.properties.lattice_constant is not None:
                diff = abs(sc.properties.lattice_constant - target)
                if diff <= tolerance:
                    results.append(sc)
        return results
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas de la base de datos."""
        total = len(self.semiconductors)
        iii_v = len(self.get_all_iii_v())
        ii_vi = len(self.get_all_ii_vi())
        stable = len(self.get_stable_semiconductors())
        
        return {
            'total': total,
            'iii_v': iii_v,
            'ii_vi': ii_vi,
            'stable': stable,
            'experimental': total - stable
        }
    
    def __len__(self) -> int:
        return len(self.semiconductors)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"SemiconductorDatabase("
                f"total={stats['total']}, "
                f"III-V={stats['iii_v']}, "
                f"II-VI={stats['ii_vi']})")


# Instancia global de la base de datos
SEMICONDUCTOR_DB = SemiconductorDatabase()
