# src/core/multi_material_config.py
"""Configuración extendida para simulaciones de múltiples materiales."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from ..config.settings import PreconvergenceConfig
from ..models.semiconductor_database import (
    BinarySemiconductor,
    SemiconductorType,
    SEMICONDUCTOR_DB
)
from .material_permutator import (
    MaterialPermutator,
    PermutationFilter,
    PermutationResult,
    MATERIAL_PERMUTATOR
)


logger = logging.getLogger(__name__)


@dataclass
class MaterialConfig:
    """Configuración para un material específico."""
    
    formula: str
    semiconductor: Optional[BinarySemiconductor] = None
    
    # Parámetros de simulación específicos del material
    lattice_constant: Optional[float] = None  # Å (usa experimental o estimado si None)
    x_position: float = 0.25  # Posición fraccionaria del catión
    
    # Parámetros computacionales (heredan de config global si None)
    cutoff: Optional[float] = None
    kmesh: Optional[Tuple[int, int, int]] = None
    basis_set: Optional[str] = None
    pseudopotential: Optional[str] = None
    
    # Prioridad de ejecución
    priority: int = 0  # Mayor = más prioritario
    enabled: bool = True
    
    def __post_init__(self):
        """Inicialización y validación."""
        if self.semiconductor is None:
            self.semiconductor = SEMICONDUCTOR_DB.get_semiconductor(self.formula)
            if self.semiconductor is None:
                logger.warning(f"Material {self.formula} no encontrado en base de datos")
        
        # Usar constante de red experimental si está disponible
        if self.lattice_constant is None and self.semiconductor:
            if self.semiconductor.properties and self.semiconductor.properties.lattice_constant:
                self.lattice_constant = self.semiconductor.properties.lattice_constant
            else:
                self.lattice_constant = self.semiconductor.estimate_lattice_constant()
                logger.info(f"{self.formula}: Usando constante de red estimada {self.lattice_constant:.3f} Å")
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'formula': self.formula,
            'lattice_constant': self.lattice_constant,
            'x_position': self.x_position,
            'cutoff': self.cutoff,
            'kmesh': self.kmesh,
            'basis_set': self.basis_set,
            'pseudopotential': self.pseudopotential,
            'priority': self.priority,
            'enabled': self.enabled
        }
    
    def __repr__(self) -> str:
        lattice_str = f"{self.lattice_constant:.3f}Å" if self.lattice_constant else "N/A"
        return f"MaterialConfig({self.formula}, a={lattice_str})"


@dataclass
class MultiMaterialConfig:
    """Configuración para simulaciones de múltiples materiales."""
    
    # Configuración base (heredada por todos los materiales)
    base_config: PreconvergenceConfig = field(default_factory=PreconvergenceConfig)
    
    # Lista de materiales a simular
    materials: List[MaterialConfig] = field(default_factory=list)
    
    # Configuración de generación automática
    auto_generate: bool = False
    generation_filter: Optional[PermutationFilter] = None
    semiconductor_types: List[SemiconductorType] = field(
        default_factory=lambda: [SemiconductorType.III_V, SemiconductorType.II_VI]
    )
    
    # Configuración de ejecución
    parallel_materials: bool = True  # Ejecutar materiales en paralelo
    max_concurrent_materials: int = 4
    
    # Directorios de salida
    output_base_dir: Path = field(default_factory=lambda: Path("results_multimaterial"))
    
    def __post_init__(self):
        """Inicialización y validación."""
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar materiales automáticamente si está habilitado
        if self.auto_generate and not self.materials:
            self._auto_generate_materials()
    
    def _auto_generate_materials(self):
        """Genera materiales automáticamente usando el permutador."""
        logger.info("Generando materiales automáticamente...")
        
        filter_config = self.generation_filter or PermutationFilter()
        
        for sem_type in self.semiconductor_types:
            if sem_type == SemiconductorType.III_V:
                result = MATERIAL_PERMUTATOR.generate_iii_v_combinations(filter_config)
            elif sem_type == SemiconductorType.II_VI:
                result = MATERIAL_PERMUTATOR.generate_ii_vi_combinations(filter_config)
            else:
                continue
            
            # Agregar semiconductores aceptados
            for semiconductor in result.filtered_combinations:
                material_config = MaterialConfig(
                    formula=semiconductor.formula,
                    semiconductor=semiconductor
                )
                self.materials.append(material_config)
        
        logger.info(f"Generados {len(self.materials)} materiales automáticamente")
    
    def add_material(self, 
                     formula: str,
                     lattice_constant: Optional[float] = None,
                     priority: int = 0,
                     **kwargs) -> MaterialConfig:
        """
        Agrega un material a la configuración.
        
        Args:
            formula: Fórmula química del material
            lattice_constant: Constante de red (usa experimental si None)
            priority: Prioridad de ejecución
            **kwargs: Parámetros adicionales para MaterialConfig
            
        Returns:
            MaterialConfig creado
        """
        material = MaterialConfig(
            formula=formula,
            lattice_constant=lattice_constant,
            priority=priority,
            **kwargs
        )
        self.materials.append(material)
        logger.info(f"Material agregado: {material}")
        return material
    
    def add_materials_from_list(self, formulas: List[str]):
        """Agrega múltiples materiales desde una lista de fórmulas."""
        for formula in formulas:
            self.add_material(formula)
    
    def add_materials_from_permutation(self, 
                                        result: PermutationResult,
                                        max_materials: Optional[int] = None):
        """
        Agrega materiales desde un resultado de permutación.
        
        Args:
            result: Resultado de permutación
            max_materials: Máximo número de materiales a agregar
        """
        semiconductors = result.filtered_combinations
        if max_materials:
            semiconductors = semiconductors[:max_materials]
        
        for semiconductor in semiconductors:
            material = MaterialConfig(
                formula=semiconductor.formula,
                semiconductor=semiconductor
            )
            self.materials.append(material)
        
        logger.info(f"Agregados {len(semiconductors)} materiales desde permutación")
    
    def remove_material(self, formula: str) -> bool:
        """Elimina un material por su fórmula."""
        initial_count = len(self.materials)
        self.materials = [m for m in self.materials if m.formula != formula]
        removed = initial_count - len(self.materials)
        if removed > 0:
            logger.info(f"Eliminado material: {formula}")
            return True
        return False
    
    def get_material(self, formula: str) -> Optional[MaterialConfig]:
        """Obtiene configuración de un material por su fórmula."""
        for material in self.materials:
            if material.formula == formula:
                return material
        return None
    
    def get_enabled_materials(self) -> List[MaterialConfig]:
        """Obtiene lista de materiales habilitados."""
        return [m for m in self.materials if m.enabled]
    
    def get_materials_by_type(self, 
                              semiconductor_type: SemiconductorType) -> List[MaterialConfig]:
        """Obtiene materiales por tipo de semiconductor."""
        return [
            m for m in self.materials
            if m.semiconductor and m.semiconductor.semiconductor_type == semiconductor_type
        ]
    
    def sort_by_priority(self):
        """Ordena materiales por prioridad (mayor primero)."""
        self.materials.sort(key=lambda m: m.priority, reverse=True)
    
    def get_output_dir(self, formula: str) -> Path:
        """Obtiene directorio de salida para un material."""
        material_dir = self.output_base_dir / formula
        material_dir.mkdir(parents=True, exist_ok=True)
        return material_dir
    
    def get_material_config_dict(self, formula: str) -> dict:
        """
        Obtiene configuración completa para un material (base + específica).
        
        Args:
            formula: Fórmula del material
            
        Returns:
            Diccionario con configuración completa
        """
        material = self.get_material(formula)
        if not material:
            raise ValueError(f"Material {formula} no encontrado")
        
        # Comenzar con configuración base
        config = self.base_config.to_dict()
        
        # Sobrescribir con parámetros específicos del material
        if material.lattice_constant:
            config['lattice_constant'] = material.lattice_constant
        
        config['x_ga'] = material.x_position
        
        if material.cutoff:
            config['cutoff_list'] = [material.cutoff]
        
        if material.kmesh:
            config['kmesh_list'] = [material.kmesh]
        
        if material.basis_set:
            config['basis_set'] = material.basis_set
        
        if material.pseudopotential:
            config['pseudopotential'] = material.pseudopotential
        
        # Configurar directorio de salida específico
        config['output_dir'] = str(self.get_output_dir(formula))
        
        return config
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas de la configuración."""
        enabled = self.get_enabled_materials()
        iii_v = self.get_materials_by_type(SemiconductorType.III_V)
        ii_vi = self.get_materials_by_type(SemiconductorType.II_VI)
        
        return {
            'total_materials': len(self.materials),
            'enabled_materials': len(enabled),
            'disabled_materials': len(self.materials) - len(enabled),
            'iii_v_materials': len(iii_v),
            'ii_vi_materials': len(ii_vi),
            'parallel_execution': self.parallel_materials,
            'max_concurrent': self.max_concurrent_materials,
            'output_dir': str(self.output_base_dir)
        }
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'base_config': self.base_config.to_dict(),
            'materials': [m.to_dict() for m in self.materials],
            'auto_generate': self.auto_generate,
            'semiconductor_types': [st.value for st in self.semiconductor_types],
            'parallel_materials': self.parallel_materials,
            'max_concurrent_materials': self.max_concurrent_materials,
            'output_base_dir': str(self.output_base_dir),
            'statistics': self.get_statistics()
        }
    
    def save_to_file(self, filepath: Path):
        """Guarda configuración a archivo YAML."""
        import yaml
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuración guardada en {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'MultiMaterialConfig':
        """Carga configuración desde archivo YAML."""
        import yaml
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruir configuración base
        base_config = PreconvergenceConfig.from_dict(data['base_config'])
        
        # Crear instancia
        config = cls(
            base_config=base_config,
            auto_generate=data.get('auto_generate', False),
            parallel_materials=data.get('parallel_materials', True),
            max_concurrent_materials=data.get('max_concurrent_materials', 4),
            output_base_dir=Path(data.get('output_base_dir', 'results_multimaterial'))
        )
        
        # Agregar materiales
        for mat_data in data.get('materials', []):
            material = MaterialConfig(
                formula=mat_data['formula'],
                lattice_constant=mat_data.get('lattice_constant'),
                x_position=mat_data.get('x_position', 0.25),
                cutoff=mat_data.get('cutoff'),
                kmesh=tuple(mat_data['kmesh']) if mat_data.get('kmesh') else None,
                basis_set=mat_data.get('basis_set'),
                pseudopotential=mat_data.get('pseudopotential'),
                priority=mat_data.get('priority', 0),
                enabled=mat_data.get('enabled', True)
            )
            config.materials.append(material)
        
        logger.info(f"Configuración cargada desde {filepath}")
        return config
    
    def __len__(self) -> int:
        return len(self.materials)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"MultiMaterialConfig("
                f"materials={stats['total_materials']}, "
                f"enabled={stats['enabled_materials']}, "
                f"III-V={stats['iii_v_materials']}, "
                f"II-VI={stats['ii_vi_materials']})")


# Funciones de conveniencia para crear configuraciones predefinidas

def create_iii_v_config(
    formulas: Optional[List[str]] = None,
    base_config: Optional[PreconvergenceConfig] = None
) -> MultiMaterialConfig:
    """
    Crea configuración para semiconductores III-V.
    
    Args:
        formulas: Lista de fórmulas (genera todos si None)
        base_config: Configuración base (usa default si None)
        
    Returns:
        MultiMaterialConfig configurado
    """
    config = MultiMaterialConfig(
        base_config=base_config or PreconvergenceConfig(),
        output_base_dir=Path("results_iii_v")
    )
    
    if formulas:
        config.add_materials_from_list(formulas)
    else:
        # Generar todos los III-V comunes
        result = MATERIAL_PERMUTATOR.generate_iii_v_combinations()
        config.add_materials_from_permutation(result)
    
    return config


def create_ii_vi_config(
    formulas: Optional[List[str]] = None,
    base_config: Optional[PreconvergenceConfig] = None
) -> MultiMaterialConfig:
    """
    Crea configuración para semiconductores II-VI.
    
    Args:
        formulas: Lista de fórmulas (genera todos si None)
        base_config: Configuración base (usa default si None)
        
    Returns:
        MultiMaterialConfig configurado
    """
    config = MultiMaterialConfig(
        base_config=base_config or PreconvergenceConfig(),
        output_base_dir=Path("results_ii_vi")
    )
    
    if formulas:
        config.add_materials_from_list(formulas)
    else:
        # Generar todos los II-VI comunes
        result = MATERIAL_PERMUTATOR.generate_ii_vi_combinations()
        config.add_materials_from_permutation(result)
    
    return config


def create_common_semiconductors_config(
    base_config: Optional[PreconvergenceConfig] = None
) -> MultiMaterialConfig:
    """
    Crea configuración con semiconductores comunes bien caracterizados.
    
    Args:
        base_config: Configuración base (usa default si None)
        
    Returns:
        MultiMaterialConfig con materiales comunes
    """
    common_materials = [
        # III-V
        'GaAs', 'GaN', 'InP', 'AlAs', 'InAs', 'GaP', 'InSb', 'AlN',
        # II-VI
        'ZnS', 'ZnSe', 'ZnTe', 'CdS', 'CdSe', 'CdTe'
    ]
    
    config = MultiMaterialConfig(
        base_config=base_config or PreconvergenceConfig(),
        output_base_dir=Path("results_common_semiconductors")
    )
    
    config.add_materials_from_list(common_materials)
    
    # Asignar prioridades (materiales más estudiados tienen mayor prioridad)
    priority_map = {
        'GaAs': 10, 'GaN': 9, 'InP': 8, 'ZnSe': 7, 'CdTe': 7,
        'AlAs': 6, 'InAs': 6, 'GaP': 5, 'ZnS': 5
    }
    
    for material in config.materials:
        material.priority = priority_map.get(material.formula, 0)
    
    config.sort_by_priority()
    
    return config
