# src/core/material_permutator.py
"""Generador automático de permutaciones de semiconductores III-V y II-VI."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from itertools import product
import logging

from ..utils.periodic_table_groups import (
    ElementData,
    PeriodicGroup,
    PERIODIC_TABLE
)
from ..models.semiconductor_database import (
    BinarySemiconductor,
    SemiconductorType,
    SemiconductorDatabase,
    SEMICONDUCTOR_DB
)


logger = logging.getLogger(__name__)


@dataclass
class PermutationFilter:
    """Filtros para permutaciones de semiconductores."""
    
    # Filtros de compatibilidad química
    max_ionic_radius_ratio: float = 2.5
    min_ionic_radius_ratio: float = 0.2
    min_electronegativity_diff: float = 0.3
    max_electronegativity_diff: float = 3.0
    
    # Filtros de elementos
    only_common_elements: bool = True
    exclude_radioactive: bool = True
    exclude_toxic: bool = False
    
    # Filtros personalizados
    custom_filters: List[Callable[[BinarySemiconductor], bool]] = field(default_factory=list)
    
    def apply(self, semiconductor: BinarySemiconductor) -> bool:
        """Aplica todos los filtros a un semiconductor."""
        
        # Filtro de razón de radios iónicos
        ratio = semiconductor.ionic_radius_ratio
        if not (self.min_ionic_radius_ratio <= ratio <= self.max_ionic_radius_ratio):
            logger.debug(f"{semiconductor.formula}: Razón de radios {ratio:.3f} fuera de rango")
            return False
        
        # Filtro de diferencia de electronegatividad
        en_diff = semiconductor.electronegativity_difference
        if not (self.min_electronegativity_diff <= en_diff <= self.max_electronegativity_diff):
            logger.debug(f"{semiconductor.formula}: Diferencia EN {en_diff:.3f} fuera de rango")
            return False
        
        # Filtro de elementos comunes
        if self.only_common_elements:
            if not (semiconductor.cation.common_in_semiconductors and 
                   semiconductor.anion.common_in_semiconductors):
                logger.debug(f"{semiconductor.formula}: Elementos no comunes en semiconductores")
                return False
        
        # Filtro de elementos radiactivos
        if self.exclude_radioactive:
            if semiconductor.cation.symbol in ['Po', 'At', 'Rn'] or \
               semiconductor.anion.symbol in ['Po', 'At', 'Rn']:
                logger.debug(f"{semiconductor.formula}: Contiene elementos radiactivos")
                return False
        
        # Filtro de elementos tóxicos
        if self.exclude_toxic:
            toxic_elements = ['Tl', 'Be', 'Cd', 'Hg']
            if semiconductor.cation.symbol in toxic_elements or \
               semiconductor.anion.symbol in toxic_elements:
                logger.debug(f"{semiconductor.formula}: Contiene elementos tóxicos")
                return False
        
        # Aplicar filtros personalizados
        for custom_filter in self.custom_filters:
            if not custom_filter(semiconductor):
                logger.debug(f"{semiconductor.formula}: Rechazado por filtro personalizado")
                return False
        
        return True


@dataclass
class PermutationResult:
    """Resultado de generación de permutaciones."""
    
    all_combinations: List[BinarySemiconductor]
    filtered_combinations: List[BinarySemiconductor]
    rejected_combinations: List[BinarySemiconductor]
    
    semiconductor_type: SemiconductorType
    filter_used: PermutationFilter
    
    @property
    def total_generated(self) -> int:
        """Total de combinaciones generadas."""
        return len(self.all_combinations)
    
    @property
    def total_accepted(self) -> int:
        """Total de combinaciones aceptadas."""
        return len(self.filtered_combinations)
    
    @property
    def total_rejected(self) -> int:
        """Total de combinaciones rechazadas."""
        return len(self.rejected_combinations)
    
    @property
    def acceptance_rate(self) -> float:
        """Tasa de aceptación (%)."""
        if self.total_generated == 0:
            return 0.0
        return (self.total_accepted / self.total_generated) * 100
    
    def get_summary(self) -> dict:
        """Obtiene resumen estadístico."""
        return {
            'type': self.semiconductor_type.value,
            'total_generated': self.total_generated,
            'total_accepted': self.total_accepted,
            'total_rejected': self.total_rejected,
            'acceptance_rate': f"{self.acceptance_rate:.1f}%",
            'accepted_formulas': [sc.formula for sc in self.filtered_combinations]
        }
    
    def __repr__(self) -> str:
        return (f"PermutationResult({self.semiconductor_type.value}: "
                f"{self.total_accepted}/{self.total_generated} accepted)")


class MaterialPermutator:
    """Generador de permutaciones de semiconductores."""
    
    def __init__(self, 
                 database: Optional[SemiconductorDatabase] = None,
                 default_filter: Optional[PermutationFilter] = None):
        """
        Inicializa el permutador.
        
        Args:
            database: Base de datos de semiconductores (usa global si None)
            default_filter: Filtro por defecto (crea uno nuevo si None)
        """
        self.database = database or SEMICONDUCTOR_DB
        self.default_filter = default_filter or PermutationFilter()
        
    def generate_iii_v_combinations(self, 
                                     filter_config: Optional[PermutationFilter] = None,
                                     cation_list: Optional[List[str]] = None,
                                     anion_list: Optional[List[str]] = None) -> PermutationResult:
        """
        Genera todas las combinaciones III-V posibles.
        
        Args:
            filter_config: Configuración de filtros (usa default si None)
            cation_list: Lista de cationes a usar (usa todos si None)
            anion_list: Lista de aniones a usar (usa todos si None)
            
        Returns:
            PermutationResult con todas las combinaciones
        """
        filter_config = filter_config or self.default_filter
        
        # Obtener elementos
        if cation_list:
            cations = {s: PERIODIC_TABLE.get_element(s) for s in cation_list}
            cations = {k: v for k, v in cations.items() if v is not None}
        else:
            cations = PERIODIC_TABLE.get_group_elements(PeriodicGroup.GROUP_III)
        
        if anion_list:
            anions = {s: PERIODIC_TABLE.get_element(s) for s in anion_list}
            anions = {k: v for k, v in anions.items() if v is not None}
        else:
            anions = PERIODIC_TABLE.get_group_elements(PeriodicGroup.GROUP_V)
        
        logger.info(f"Generando combinaciones III-V: {len(cations)} cationes × {len(anions)} aniones")
        
        # Generar todas las combinaciones
        all_combinations = []
        for cation_data, anion_data in product(cations.values(), anions.values()):
            try:
                semiconductor = BinarySemiconductor(
                    cation=cation_data,
                    anion=anion_data,
                    semiconductor_type=SemiconductorType.III_V
                )
                all_combinations.append(semiconductor)
            except ValueError as e:
                logger.warning(f"Error creando {cation_data.symbol}{anion_data.symbol}: {e}")
        
        # Aplicar filtros
        filtered = []
        rejected = []
        for sc in all_combinations:
            if filter_config.apply(sc):
                filtered.append(sc)
                # Agregar a la base de datos si no existe
                if sc.formula not in self.database.semiconductors:
                    self.database.add_semiconductor(sc)
            else:
                rejected.append(sc)
        
        logger.info(f"Combinaciones III-V: {len(filtered)} aceptadas, {len(rejected)} rechazadas")
        
        return PermutationResult(
            all_combinations=all_combinations,
            filtered_combinations=filtered,
            rejected_combinations=rejected,
            semiconductor_type=SemiconductorType.III_V,
            filter_used=filter_config
        )
    
    def generate_ii_vi_combinations(self,
                                     filter_config: Optional[PermutationFilter] = None,
                                     cation_list: Optional[List[str]] = None,
                                     anion_list: Optional[List[str]] = None) -> PermutationResult:
        """
        Genera todas las combinaciones II-VI posibles.
        
        Args:
            filter_config: Configuración de filtros (usa default si None)
            cation_list: Lista de cationes a usar (usa todos si None)
            anion_list: Lista de aniones a usar (usa todos si None)
            
        Returns:
            PermutationResult con todas las combinaciones
        """
        filter_config = filter_config or self.default_filter
        
        # Obtener elementos
        if cation_list:
            cations = {s: PERIODIC_TABLE.get_element(s) for s in cation_list}
            cations = {k: v for k, v in cations.items() if v is not None}
        else:
            cations = PERIODIC_TABLE.get_group_elements(PeriodicGroup.GROUP_II)
        
        if anion_list:
            anions = {s: PERIODIC_TABLE.get_element(s) for s in anion_list}
            anions = {k: v for k, v in anions.items() if v is not None}
        else:
            anions = PERIODIC_TABLE.get_group_elements(PeriodicGroup.GROUP_VI)
        
        logger.info(f"Generando combinaciones II-VI: {len(cations)} cationes × {len(anions)} aniones")
        
        # Generar todas las combinaciones
        all_combinations = []
        for cation_data, anion_data in product(cations.values(), anions.values()):
            try:
                semiconductor = BinarySemiconductor(
                    cation=cation_data,
                    anion=anion_data,
                    semiconductor_type=SemiconductorType.II_VI
                )
                all_combinations.append(semiconductor)
            except ValueError as e:
                logger.warning(f"Error creando {cation_data.symbol}{anion_data.symbol}: {e}")
        
        # Aplicar filtros
        filtered = []
        rejected = []
        for sc in all_combinations:
            if filter_config.apply(sc):
                filtered.append(sc)
                # Agregar a la base de datos si no existe
                if sc.formula not in self.database.semiconductors:
                    self.database.add_semiconductor(sc)
            else:
                rejected.append(sc)
        
        logger.info(f"Combinaciones II-VI: {len(filtered)} aceptadas, {len(rejected)} rechazadas")
        
        return PermutationResult(
            all_combinations=all_combinations,
            filtered_combinations=filtered,
            rejected_combinations=rejected,
            semiconductor_type=SemiconductorType.II_VI,
            filter_used=filter_config
        )
    
    def generate_all_combinations(self,
                                   filter_config: Optional[PermutationFilter] = None) -> Tuple[PermutationResult, PermutationResult]:
        """
        Genera todas las combinaciones III-V y II-VI.
        
        Args:
            filter_config: Configuración de filtros
            
        Returns:
            Tupla (resultado_III_V, resultado_II_VI)
        """
        logger.info("Generando todas las combinaciones de semiconductores...")
        
        iii_v_result = self.generate_iii_v_combinations(filter_config)
        ii_vi_result = self.generate_ii_vi_combinations(filter_config)
        
        total_accepted = iii_v_result.total_accepted + ii_vi_result.total_accepted
        total_generated = iii_v_result.total_generated + ii_vi_result.total_generated
        
        logger.info(f"Total: {total_accepted}/{total_generated} semiconductores aceptados")
        
        return iii_v_result, ii_vi_result
    
    def generate_custom_combinations(self,
                                      cations: List[str],
                                      anions: List[str],
                                      semiconductor_type: SemiconductorType,
                                      filter_config: Optional[PermutationFilter] = None) -> PermutationResult:
        """
        Genera combinaciones personalizadas.
        
        Args:
            cations: Lista de símbolos de cationes
            anions: Lista de símbolos de aniones
            semiconductor_type: Tipo de semiconductor
            filter_config: Configuración de filtros
            
        Returns:
            PermutationResult con las combinaciones
        """
        if semiconductor_type == SemiconductorType.III_V:
            return self.generate_iii_v_combinations(
                filter_config=filter_config,
                cation_list=cations,
                anion_list=anions
            )
        elif semiconductor_type == SemiconductorType.II_VI:
            return self.generate_ii_vi_combinations(
                filter_config=filter_config,
                cation_list=cations,
                anion_list=anions
            )
        else:
            raise ValueError(f"Tipo de semiconductor no soportado: {semiconductor_type}")
    
    def find_lattice_matched_pairs(self,
                                    target_lattice: float,
                                    tolerance: float = 0.1,
                                    semiconductor_type: Optional[SemiconductorType] = None) -> List[BinarySemiconductor]:
        """
        Encuentra semiconductores con constante de red similar.
        
        Args:
            target_lattice: Constante de red objetivo (Å)
            tolerance: Tolerancia (Å)
            semiconductor_type: Tipo de semiconductor (None = todos)
            
        Returns:
            Lista de semiconductores compatibles
        """
        results = []
        
        for sc in self.database.semiconductors.values():
            # Filtrar por tipo si se especifica
            if semiconductor_type and sc.semiconductor_type != semiconductor_type:
                continue
            
            # Usar constante de red experimental o estimada
            if sc.properties and sc.properties.lattice_constant:
                lattice = sc.properties.lattice_constant
            else:
                lattice = sc.estimate_lattice_constant()
            
            # Verificar tolerancia
            if abs(lattice - target_lattice) <= tolerance:
                results.append(sc)
        
        # Ordenar por diferencia de constante de red
        results.sort(key=lambda sc: abs(
            (sc.properties.lattice_constant if sc.properties and sc.properties.lattice_constant 
             else sc.estimate_lattice_constant()) - target_lattice
        ))
        
        return results
    
    def suggest_heterostructures(self,
                                  base_material: str,
                                  max_lattice_mismatch: float = 0.05) -> List[Tuple[BinarySemiconductor, float]]:
        """
        Sugiere materiales para heteroestructuras.
        
        Args:
            base_material: Fórmula del material base
            max_lattice_mismatch: Máximo desajuste de red (fracción)
            
        Returns:
            Lista de tuplas (semiconductor, mismatch)
        """
        base = self.database.get_semiconductor(base_material)
        if not base:
            raise ValueError(f"Material base {base_material} no encontrado")
        
        if base.properties and base.properties.lattice_constant:
            base_lattice = base.properties.lattice_constant
        else:
            base_lattice = base.estimate_lattice_constant()
        
        suggestions = []
        
        for sc in self.database.semiconductors.values():
            # No sugerir el mismo material
            if sc.formula == base_material:
                continue
            
            # Obtener constante de red
            if sc.properties and sc.properties.lattice_constant:
                lattice = sc.properties.lattice_constant
            else:
                lattice = sc.estimate_lattice_constant()
            
            # Calcular desajuste
            mismatch = abs(lattice - base_lattice) / base_lattice
            
            if mismatch <= max_lattice_mismatch:
                suggestions.append((sc, mismatch))
        
        # Ordenar por desajuste
        suggestions.sort(key=lambda x: x[1])
        
        return suggestions


# Instancia global del permutador
MATERIAL_PERMUTATOR = MaterialPermutator()


# Funciones de conveniencia
def generate_all_iii_v(filter_config: Optional[PermutationFilter] = None) -> PermutationResult:
    """Genera todas las combinaciones III-V."""
    return MATERIAL_PERMUTATOR.generate_iii_v_combinations(filter_config)


def generate_all_ii_vi(filter_config: Optional[PermutationFilter] = None) -> PermutationResult:
    """Genera todas las combinaciones II-VI."""
    return MATERIAL_PERMUTATOR.generate_ii_vi_combinations(filter_config)


def generate_all_semiconductors(filter_config: Optional[PermutationFilter] = None) -> Tuple[PermutationResult, PermutationResult]:
    """Genera todas las combinaciones de semiconductores."""
    return MATERIAL_PERMUTATOR.generate_all_combinations(filter_config)
