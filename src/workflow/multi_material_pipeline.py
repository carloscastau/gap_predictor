# src/workflow/multi_material_pipeline.py
"""Pipeline integrado para múltiples materiales con ejecución optimizada."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ..config.settings import PreconvergenceConfig
from ..core.material_permutator import (
    MaterialPermutator,
    PermutationFilter,
    PermutationResult,
    MATERIAL_PERMUTATOR
)
from ..core.multi_material_config import (
    MultiMaterialConfig,
    MaterialConfig,
    create_common_semiconductors_config
)
from ..models.semiconductor_database import (
    BinarySemiconductor,
    SemiconductorType,
    SEMICONDUCTOR_DB
)
from .pipeline import PreconvergencePipeline, PipelineResult
from .batch_processor import BatchProcessor
from ..utils.logging import StructuredLogger


logger = logging.getLogger(__name__)


@dataclass
class MaterialExecutionResult:
    """Resultado de ejecución de un material individual."""
    formula: str
    success: bool
    pipeline_result: Optional[PipelineResult] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    optimal_cutoff: Optional[float] = None
    optimal_kmesh: Optional[tuple] = None
    optimal_lattice_constant: Optional[float] = None
    
    def get_summary(self) -> dict:
        """Obtiene resumen del resultado."""
        return {
            'formula': self.formula,
            'success': self.success,
            'execution_time': self.execution_time,
            'stages_completed': self.stages_completed,
            'optimal_cutoff': self.optimal_cutoff,
            'optimal_kmesh': self.optimal_kmesh,
            'optimal_lattice_constant': self.optimal_lattice_constant,
            'error_message': self.error_message
        }


@dataclass
class CampaignResult:
    """Resultado completo de una campaña multimaterial."""
    materials_executed: int
    materials_successful: int
    materials_failed: int
    total_execution_time: float
    individual_results: List[MaterialExecutionResult]
    campaign_config: MultiMaterialConfig
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito de la campaña."""
        if self.materials_executed == 0:
            return 0.0
        return (self.materials_successful / self.materials_executed) * 100
    
    @property
    def average_execution_time(self) -> float:
        """Tiempo promedio de ejecución por material."""
        if self.materials_executed == 0:
            return 0.0
        return self.total_execution_time / self.materials_executed
    
    def get_successful_materials(self) -> List[str]:
        """Obtiene lista de materiales exitosos."""
        return [
            r.formula for r in self.individual_results 
            if r.success
        ]
    
    def get_failed_materials(self) -> List[str]:
        """Obtiene lista de materiales fallidos."""
        return [
            r.formula for r in self.individual_results 
            if not r.success
        ]
    
    def get_consolidated_results(self) -> dict:
        """Obtiene resultados consolidados para análisis."""
        successful_results = [r for r in self.individual_results if r.success]
        
        if not successful_results:
            return {'error': 'No successful materials to consolidate'}
        
        # Recopilar parámetros óptimos
        cutoffs = [r.optimal_cutoff for r in successful_results if r.optimal_cutoff]
        kmeshes = [r.optimal_kmesh for r in successful_results if r.optimal_kmesh]
        lattices = [r.optimal_lattice_constant for r in successful_results if r.optimal_lattice_constant]
        
        return {
            'campaign_summary': {
                'total_materials': self.materials_executed,
                'successful': self.materials_successful,
                'failed': self.materials_failed,
                'success_rate': self.success_rate,
                'total_time': self.total_execution_time,
                'average_time': self.average_execution_time
            },
            'optimal_parameters': {
                'cutoffs': {
                    'values': cutoffs,
                    'min': min(cutoffs) if cutoffs else None,
                    'max': max(cutoffs) if cutoffs else None,
                    'average': sum(cutoffs) / len(cutoffs) if cutoffs else None
                },
                'lattice_constants': {
                    'values': lattices,
                    'min': min(lattices) if lattices else None,
                    'max': max(lattices) if lattices else None,
                    'average': sum(lattices) / len(lattices) if lattices else None
                }
            },
            'materials_by_success': {
                'successful': self.get_successful_materials(),
                'failed': self.get_failed_materials()
            }
        }


class MultiMaterialPipeline:
    """Pipeline principal para ejecutar preconvergencia en múltiples materiales."""
    
    def __init__(self, 
                 config: Optional[MultiMaterialConfig] = None,
                 enable_monitoring: bool = True):
        """
        Inicializa el pipeline multimaterial.
        
        Args:
            config: Configuración multimaterial (usa default si None)
            enable_monitoring: Habilitar monitoreo de producción
        """
        self.config = config or create_common_semiconductors_config()
        self.logger = StructuredLogger("MultiMaterialPipeline")
        self.batch_processor = BatchProcessor(
            max_concurrent=self.config.max_concurrent_materials,
            enable_progress_tracking=True
        )
        
        # Monitoreo
        self.monitoring_enabled = enable_monitoring
        self.monitor = None
        if enable_monitoring:
            # TODO: Implementar monitor de producción cuando esté disponible
            self.logger.warning("Production monitoring not available in this version")
        
        # Cache de pipelines individuales
        self._pipeline_cache: Dict[str, PreconvergencePipeline] = {}
        
        self.logger.info(f"MultiMaterialPipeline inicializado con {len(self.config.materials)} materiales")
    
    def add_materials_from_list(self, formulas: List[str]):
        """Agrega materiales desde una lista de fórmulas."""
        self.config.add_materials_from_list(formulas)
        self.logger.info(f"Agregados {len(formulas)} materiales: {formulas}")
    
    def add_materials_from_permutation(self, result: PermutationResult, max_materials: Optional[int] = None):
        """Agrega materiales desde resultado de permutación."""
        initial_count = len(self.config.materials)
        self.config.add_materials_from_permutation(result, max_materials)
        added_count = len(self.config.materials) - initial_count
        self.logger.info(f"Agregados {added_count} materiales desde permutación")
    
    def set_parallel_workers(self, max_workers: int):
        """Configura el número de workers paralelos."""
        self.config.max_concurrent_materials = max_workers
        self.batch_processor.max_concurrent = max_workers
        self.logger.info(f"Configurados {max_workers} workers paralelos")
    
    def enable_parallel_execution(self, enabled: bool = True):
        """Habilita/deshabilita ejecución paralela de materiales."""
        self.config.parallel_materials = enabled
        self.batch_processor.parallel_enabled = enabled
        status = "habilitada" if enabled else "deshabilitada"
        self.logger.info(f"Ejecución paralela {status}")
    
    def get_material_pipeline(self, formula: str) -> PreconvergencePipeline:
        """
        Obtiene o crea pipeline para un material específico.
        
        Args:
            formula: Fórmula del material
            
        Returns:
            Pipeline configurado para el material
        """
        if formula in self._pipeline_cache:
            return self._pipeline_cache[formula]
        
        # Obtener configuración específica del material
        material_config = self.config.get_material(formula)
        if not material_config:
            raise ValueError(f"Material {formula} no encontrado en configuración")
        
        # Crear configuración de preconvergencia específica
        config_dict = self.config.get_material_config_dict(formula)
        material_pipeline_config = PreconvergenceConfig.from_dict(config_dict)
        
        # Crear pipeline
        pipeline = PreconvergencePipeline(
            config=material_pipeline_config,
            enable_monitoring=self.monitoring_enabled
        )
        
        # Cachear pipeline
        self._pipeline_cache[formula] = pipeline
        
        return pipeline
    
    def validate_materials(self) -> Dict[str, Any]:
        """
        Valida todos los materiales antes de ejecutar.
        
        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            'valid_materials': [],
            'invalid_materials': [],
            'warnings': [],
            'total_valid': 0,
            'total_invalid': 0
        }
        
        for material in self.config.get_enabled_materials():
            material_validation = self._validate_single_material(material)
            
            if material_validation['valid']:
                validation_results['valid_materials'].append(material.formula)
                validation_results['total_valid'] += 1
            else:
                validation_results['invalid_materials'].append({
                    'formula': material.formula,
                    'errors': material_validation['errors']
                })
                validation_results['total_invalid'] += 1
            
            validation_results['warnings'].extend(material_validation['warnings'])
        
        self.logger.info(f"Validación completada: {validation_results['total_valid']} válidos, "
                        f"{validation_results['total_invalid']} inválidos")
        
        return validation_results
    
    def _validate_single_material(self, material: MaterialConfig) -> dict:
        """Valida un material individual."""
        errors = []
        warnings = []
        
        # Verificar que el material existe en la base de datos
        if not material.semiconductor:
            warnings.append(f"Material {material.formula} no encontrado en base de datos")
        
        # Verificar constante de red
        if material.lattice_constant is None:
            errors.append(f"Constante de red no disponible para {material.formula}")
        elif not (3.0 <= material.lattice_constant <= 8.0):
            warnings.append(f"Constante de red {material.lattice_constant} parece atípica para {material.formula}")
        
        # Verificar parámetros computacionales
        if material.cutoff and (material.cutoff < 200 or material.cutoff > 1000):
            warnings.append(f"Cutoff {material.cutoff} puede ser atípico para {material.formula}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def execute_single_material(self, formula: str, resume_from: Optional[str] = None) -> MaterialExecutionResult:
        """
        Ejecuta preconvergencia para un material individual.
        
        Args:
            formula: Fórmula del material
            resume_from: Stage desde donde reanudar (None = desde el inicio)
            
        Returns:
            MaterialExecutionResult con el resultado
        """
        start_time = time.perf_counter()
        
        try:
            self.logger.info(f"Iniciando ejecución de material: {formula}")
            
            # Obtener pipeline específico
            pipeline = self.get_material_pipeline(formula)
            
            # Ejecutar pipeline
            pipeline_result = await pipeline.execute(resume_from)
            
            # Extraer parámetros óptimos
            optimal_cutoff = None
            optimal_kmesh = None
            optimal_lattice_constant = None
            
            if pipeline_result.success and pipeline_result.results:
                # Obtener cutoff óptimo
                if 'cutoff' in pipeline_result.results:
                    optimal_cutoff = pipeline_result.results['cutoff'].data.get('optimal_cutoff')
                
                # Obtener kmesh óptimo
                if 'kmesh' in pipeline_result.results:
                    optimal_kmesh = pipeline_result.results['kmesh'].data.get('optimal_kmesh')
                
                # Obtener lattice óptimo
                if 'lattice' in pipeline_result.results:
                    optimal_lattice_constant = pipeline_result.results['lattice'].data.get('optimal_constant')
            
            execution_time = time.perf_counter() - start_time
            
            result = MaterialExecutionResult(
                formula=formula,
                success=pipeline_result.success,
                pipeline_result=pipeline_result,
                execution_time=execution_time,
                stages_completed=list(pipeline_result.results.keys()) if pipeline_result.success else [],
                optimal_cutoff=optimal_cutoff,
                optimal_kmesh=optimal_kmesh,
                optimal_lattice_constant=optimal_lattice_constant
            )
            
            if pipeline_result.success:
                self.logger.info(f"Material {formula} completado exitosamente en {execution_time:.2f}s")
            else:
                self.logger.warning(f"Material {formula} falló: {pipeline_result.error_message}")
            
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            self.logger.error(f"Error ejecutando material {formula}: {e}")
            
            return MaterialExecutionResult(
                formula=formula,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def run_preconvergence_campaign(self, 
                                        materials: Optional[List[str]] = None,
                                        resume_from: Optional[str] = None,
                                        progress_callback: Optional[Callable[[int, int], None]] = None) -> CampaignResult:
        """
        Ejecuta una campaña completa de preconvergencia para múltiples materiales.
        
        Args:
            materials: Lista de materiales a ejecutar (None = todos habilitados)
            resume_from: Stage desde donde reanudar
            progress_callback: Callback de progreso (completed, total)
            
        Returns:
            CampaignResult con resultados consolidados
        """
        start_time = time.perf_counter()
        
        # Seleccionar materiales
        if materials:
            target_materials = [m for m in materials if self.config.get_material(m)]
            if len(target_materials) != len(materials):
                missing = set(materials) - set(target_materials)
                self.logger.warning(f"Materiales no encontrados: {missing}")
        else:
            target_materials = [m.formula for m in self.config.get_enabled_materials()]
        
        if not target_materials:
            raise ValueError("No hay materiales para ejecutar")
        
        self.logger.info(f"Iniciando campaña para {len(target_materials)} materiales")
        
        # Validar materiales
        validation = self.validate_materials()
        if validation['total_invalid'] > 0:
            self.logger.error(f"Materiales inválidos encontrados: {validation['invalid_materials']}")
            raise ValueError(f"Materiales inválidos: {validation['invalid_materials']}")
        
        # Configurar monitoreo si está habilitado
        if self.monitoring_enabled and self.monitor:
            self.monitor.start_campaign_monitoring(len(target_materials))
        
        # Ejecutar materiales
        if self.config.parallel_materials:
            results = await self._execute_parallel_materials(
                target_materials, 
                resume_from, 
                progress_callback
            )
        else:
            results = await self._execute_sequential_materials(
                target_materials, 
                resume_from, 
                progress_callback
            )
        
        total_time = time.perf_counter() - start_time
        
        # Crear resultado consolidado
        campaign_result = CampaignResult(
            materials_executed=len(target_materials),
            materials_successful=sum(1 for r in results if r.success),
            materials_failed=sum(1 for r in results if not r.success),
            total_execution_time=total_time,
            individual_results=results,
            campaign_config=self.config
        )
        
        # Detener monitoreo
        if self.monitoring_enabled and self.monitor:
            self.monitor.stop_campaign_monitoring()
        
        # Log resumen final
        self.logger.info(f"Campaña completada: {campaign_result.materials_successful}/"
                        f"{campaign_result.materials_executed} exitosos "
                        f"({campaign_result.success_rate:.1f}% éxito)")
        
        return campaign_result
    
    async def _execute_parallel_materials(self, 
                                        materials: List[str], 
                                        resume_from: Optional[str],
                                        progress_callback: Optional[Callable]) -> List[MaterialExecutionResult]:
        """Ejecuta materiales en paralelo usando BatchProcessor."""
        self.logger.info(f"Ejecutando {len(materials)} materiales en paralelo")
        
        # Configurar tareas para batch processor
        async def execute_material_task(formula: str) -> MaterialExecutionResult:
            return await self.execute_single_material(formula, resume_from)
        
        # Ejecutar con batch processor
        results = await self.batch_processor.process_batch(
            items=materials,
            process_func=execute_material_task,
            progress_callback=progress_callback
        )
        
        return results
    
    async def _execute_sequential_materials(self, 
                                          materials: List[str], 
                                          resume_from: Optional[str],
                                          progress_callback: Optional[Callable]) -> List[MaterialExecutionResult]:
        """Ejecuta materiales secuencialmente."""
        self.logger.info(f"Ejecutando {len(materials)} materiales secuencialmente")
        
        results = []
        for i, formula in enumerate(materials):
            try:
                result = await self.execute_single_material(formula, resume_from)
                results.append(result)
                
                # Callback de progreso
                if progress_callback:
                    progress_callback(i + 1, len(materials))
                
                # Pausa pequeña entre materiales para evitar sobrecarga
                if i < len(materials) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error en material {formula}: {e}")
                results.append(MaterialExecutionResult(
                    formula=formula,
                    success=False,
                    error_message=str(e)
                ))
                
                if progress_callback:
                    progress_callback(i + 1, len(materials))
        
        return results
    
    def save_campaign_results(self, campaign_result: CampaignResult, filepath: Path):
        """Guarda resultados de campaña a archivo."""
        results_data = {
            'campaign_summary': {
                'materials_executed': campaign_result.materials_executed,
                'materials_successful': campaign_result.materials_successful,
                'materials_failed': campaign_result.materials_failed,
                'success_rate': campaign_result.success_rate,
                'total_execution_time': campaign_result.total_execution_time,
                'average_execution_time': campaign_result.average_execution_time
            },
            'individual_results': [r.get_summary() for r in campaign_result.individual_results],
            'consolidated_results': campaign_result.get_consolidated_results(),
            'config': self.config.to_dict(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Resultados guardados en {filepath}")
    
    def get_campaign_progress(self) -> Dict[str, Any]:
        """Obtiene progreso actual de la campaña."""
        return self.batch_processor.get_progress_status()
    
    def stop_campaign(self):
        """Detiene la campaña en ejecución."""
        self.batch_processor.stop_processing()
        self.logger.info("Campaña detenida por solicitud del usuario")


# Funciones de conveniencia para crear campañas predefinidas

async def run_common_semiconductors_campaign(
    materials: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: int = 4
) -> CampaignResult:
    """
    Ejecuta campaña para semiconductores comunes.
    
    Args:
        materials: Lista de materiales (usa predefinidos si None)
        parallel: Ejecutar en paralelo
        max_workers: Número máximo de workers
        
    Returns:
        CampaignResult con resultados
    """
    # Crear pipeline con semiconductores comunes
    pipeline = MultiMaterialPipeline()
    if materials:
        pipeline.add_materials_from_list(materials)
    
    # Configurar ejecución
    pipeline.enable_parallel_execution(parallel)
    pipeline.set_parallel_workers(max_workers)
    
    # Ejecutar campaña
    return await pipeline.run_preconvergence_campaign()


async def run_custom_materials_campaign(
    materials: List[str],
    parallel: bool = True,
    max_workers: int = 4
) -> CampaignResult:
    """
    Ejecuta campaña para materiales personalizados.
    
    Args:
        materials: Lista de materiales personalizados
        parallel: Ejecutar en paralelo
        max_workers: Número máximo de workers
        
    Returns:
        CampaignResult con resultados
    """
    pipeline = MultiMaterialPipeline()
    pipeline.add_materials_from_list(materials)
    
    pipeline.enable_parallel_execution(parallel)
    pipeline.set_parallel_workers(max_workers)
    
    return await pipeline.run_preconvergence_campaign()


async def run_generated_materials_campaign(
    semiconductor_types: List[SemiconductorType] = None,
    max_materials: int = 10,
    parallel: bool = True,
    max_workers: int = 4
) -> CampaignResult:
    """
    Ejecuta campaña para materiales generados automáticamente.
    
    Args:
        semiconductor_types: Tipos de semiconductores a generar
        max_materials: Máximo número de materiales
        parallel: Ejecutar en paralelo
        max_workers: Número máximo de workers
        
    Returns:
        CampaignResult con resultados
    """
    if semiconductor_types is None:
        semiconductor_types = [SemiconductorType.III_V, SemiconductorType.II_VI]
    
    # Generar materiales
    filter_config = PermutationFilter()
    materials = []
    
    for sem_type in semiconductor_types:
        if sem_type == SemiconductorType.III_V:
            result = MATERIAL_PERMUTATOR.generate_iii_v_combinations(filter_config)
        elif sem_type == SemiconductorType.II_VI:
            result = MATERIAL_PERMUTATOR.generate_ii_vi_combinations(filter_config)
        else:
            continue
        
        materials.extend([sc.formula for sc in result.filtered_combinations[:max_materials]])
        
        if len(materials) >= max_materials:
            break
    
    # Crear pipeline y ejecutar
    pipeline = MultiMaterialPipeline()
    pipeline.add_materials_from_list(materials[:max_materials])
    
    pipeline.enable_parallel_execution(parallel)
    pipeline.set_parallel_workers(max_workers)
    
    return await pipeline.run_preconvergence_campaign()