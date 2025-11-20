# src/workflow/pipeline.py
"""Pipeline principal con stages modulares e independientes."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..config.settings import PreconvergenceConfig
from ..core.calculator import DFTCalculator, CellParameters
from ..core.optimizer import LatticeOptimizer, ConvergenceAnalyzer
from ..core.parallel import TaskScheduler, CalculationTask
from ..workflow.checkpoint import CheckpointManager
from ..utils.logging import StructuredLogger
# from ..utils.production_monitor import create_production_monitor  # Archivo problemático eliminado

# Importaciones para integración multimaterial
try:
    from .multi_material_pipeline import (
        MultiMaterialPipeline, 
        CampaignResult,
        MaterialExecutionResult,
        run_preconvergence_campaign
    )
    MULTIMATERIAL_AVAILABLE = True
except ImportError:
    MULTIMATERIAL_AVAILABLE = False


@dataclass
class StageResult:
    """Resultado estándar de un stage."""
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    duration: float
    timestamp: str
    stage_name: str


@dataclass
class PipelineResult:
    """Resultado completo del pipeline."""
    results: Dict[str, StageResult]
    config: PreconvergenceConfig
    total_duration: float
    success: bool
    error_message: Optional[str] = None


class PipelineStage:
    """Interface común para todos los stages del pipeline."""

    def __init__(self, config: PreconvergenceConfig, name: str):
        self.config = config
        self.name = name
        self.logger = StructuredLogger(f"Stage.{name}")

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta el stage y retorna resultado."""
        start_time = time.perf_counter()

        try:
            self.logger.info(f"Starting stage {self.name}")

            # Validar entradas
            if not self.validate_inputs(previous_results):
                raise ValueError(f"Invalid inputs for stage {self.name}")

            # Ejecutar lógica específica del stage
            result_data = await self._execute_stage(previous_results)

            duration = time.perf_counter() - start_time

            result = StageResult(
                success=True,
                data=result_data,
                metrics=self._collect_metrics(),
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                stage_name=self.name
            )

            self.logger.info(f"Stage {self.name} completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(f"Stage {self.name} failed after {duration:.2f}s: {e}")

            return StageResult(
                success=False,
                data={},
                metrics={'error': str(e)},
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                stage_name=self.name
            )

    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        """Valida que las entradas del stage sean correctas."""
        required_stages = self.get_dependencies()
        return all(stage in previous_results for stage in required_stages)

    def get_dependencies(self) -> List[str]:
        """Retorna lista de stages de los que depende este stage."""
        return []

    async def _execute_stage(self, previous_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Implementación específica del stage."""
        raise NotImplementedError

    def _collect_metrics(self) -> Dict[str, Any]:
        """Recopila métricas del stage."""
        return {}


class CutoffConvergenceStage(PipelineStage):
    """Stage de convergencia de cutoff."""

    def __init__(self, config: PreconvergenceConfig):
        super().__init__(config, "cutoff")

    def get_dependencies(self) -> List[str]:
        return []  # No dependencies

    async def _execute_stage(self, previous_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Ejecuta convergencia de cutoff."""
        calculator = DFTCalculator(self.config)
        analyzer = ConvergenceAnalyzer(self.config)

        # Crear tareas para diferentes cutoffs
        tasks = []
        for cutoff in self.config.cutoff_list:
            task = CalculationTask(
                task_id=f"cutoff_{cutoff}",
                cutoff=cutoff,
                kmesh=self.config.kmesh_list[0],  # Usar primera malla k
                lattice_constant=self.config.lattice_constant,
                x_ga=self.config.x_ga,
                basis=self.config.basis_set,
                pseudo=self.config.pseudopotential,
                xc=self.config.xc_functional,
                sigma_ha=self.config.sigma_ha,
                conv_tol=1e-8
            )
            tasks.append(task)

        # Ejecutar cálculos
        scheduler = TaskScheduler(self.config)
        results = await scheduler.schedule_tasks(tasks)

        # Convertir a puntos de convergencia
        convergence_points = []
        for task_id, result in results.items():
            if result.success:
                cutoff = float(task_id.split('_')[1])
                convergence_points.append({
                    'parameter': cutoff,
                    'energy': result.energy,
                    'error': None
                })

        # Analizar convergencia
        from ..core.optimizer import ConvergencePoint
        points = [ConvergencePoint(**p) for p in convergence_points]
        convergence_result = analyzer.analyze_cutoff_convergence(points)

        return {
            'cutoff_values': self.config.cutoff_list,
            'energies': [r.energy for r in results.values() if r.success],
            'optimal_cutoff': convergence_result.optimal_value,
            'converged': convergence_result.converged,
            'threshold_used': convergence_result.convergence_threshold,
            'points_analyzed': convergence_result.points_used
        }


class KMeshConvergenceStage(PipelineStage):
    """Stage de convergencia de k-mesh."""

    def __init__(self, config: PreconvergenceConfig):
        super().__init__(config, "kmesh")

    def get_dependencies(self) -> List[str]:
        return ["cutoff"]  # Depende del cutoff óptimo

    async def _execute_stage(self, previous_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Ejecuta convergencia de k-mesh."""
        cutoff_result = previous_results["cutoff"]
        optimal_cutoff = cutoff_result.data.get("optimal_cutoff", self.config.cutoff_list[-1])

        calculator = DFTCalculator(self.config)
        analyzer = ConvergenceAnalyzer(self.config)

        # Crear tareas para diferentes k-mesh
        tasks = []
        for kmesh in self.config.kmesh_list:
            task = CalculationTask(
                task_id=f"kmesh_{kmesh[0]}x{kmesh[1]}x{kmesh[2]}",
                cutoff=optimal_cutoff,
                kmesh=kmesh,
                lattice_constant=self.config.lattice_constant,
                x_ga=self.config.x_ga,
                basis=self.config.basis_set,
                pseudo=self.config.pseudopotential,
                xc=self.config.xc_functional,
                sigma_ha=self.config.sigma_ha,
                conv_tol=1e-8
            )
            tasks.append(task)

        # Ejecutar cálculos
        scheduler = TaskScheduler(self.config)
        results = await scheduler.schedule_tasks(tasks)

        # Convertir a puntos de convergencia
        convergence_points = []
        for task_id, result in results.items():
            if result.success:
                # Extraer kmesh del task_id
                kmesh_str = task_id.split('_')[1]
                kx, ky, kz = map(int, kmesh_str.split('x'))
                nkpts = kx * ky * kz
                convergence_points.append({
                    'parameter': nkpts,  # Usar número total de k-points
                    'energy': result.energy,
                    'error': None,
                    'kmesh': (kx, ky, kz)
                })

        # Analizar convergencia
        from ..core.optimizer import ConvergencePoint
        points = [ConvergencePoint(**p) for p in convergence_points]
        convergence_result = analyzer.analyze_kmesh_convergence(points)

        # Encontrar kmesh correspondiente al óptimo
        optimal_kmesh = None
        for point in convergence_points:
            if point['parameter'] == convergence_result.optimal_value:
                optimal_kmesh = point['kmesh']
                break

        if optimal_kmesh is None:
            optimal_kmesh = self.config.kmesh_list[-1]  # Fallback

        return {
            'kmesh_values': self.config.kmesh_list,
            'energies': [r.energy for r in results.values() if r.success],
            'optimal_kmesh': optimal_kmesh,
            'optimal_nkpts': convergence_result.optimal_value,
            'converged': convergence_result.converged,
            'threshold_used': convergence_result.convergence_threshold,
            'points_analyzed': convergence_result.points_used
        }


class LatticeOptimizationStage(PipelineStage):
    """Stage de optimización de parámetro de red."""

    def __init__(self, config: PreconvergenceConfig):
        super().__init__(config, "lattice")

    def get_dependencies(self) -> List[str]:
        return ["cutoff", "kmesh"]  # Depende de cutoff y kmesh óptimos

    async def _execute_stage(self, previous_results: Dict[str, StageResult]) -> Dict[str, Any]:
        """Ejecuta optimización de lattice."""
        cutoff_result = previous_results["cutoff"]
        kmesh_result = previous_results["kmesh"]

        optimal_cutoff = cutoff_result.data.get("optimal_cutoff", self.config.cutoff_list[-1])
        optimal_kmesh = kmesh_result.data.get("optimal_kmesh", self.config.kmesh_list[-1])

        optimizer = LatticeOptimizer(self.config)

        # Función de energía para optimización
        async def energy_func(a: float) -> float:
            calculator = DFTCalculator(self.config)
            cell_params = CellParameters(
                lattice_constant=a,
                x_ga=self.config.x_ga,
                cutoff=optimal_cutoff,
                kmesh=optimal_kmesh,
                basis=self.config.basis_set,
                pseudo=self.config.pseudopotential,
                xc=self.config.xc_functional,
                sigma_ha=self.config.sigma_ha,
                conv_tol=1e-8
            )

            result = await calculator.calculate_energy(cell_params)
            return result.energy if result.converged else float('nan')

        # Ejecutar optimización
        optimization_result = await optimizer.optimize_lattice_constant(
            energy_func,
            a_range=(5.4, 5.8),
            n_points=7
        )

        return optimization_result


class PreconvergencePipeline:
    """Pipeline principal con stages modulares e independientes."""

    def __init__(self, config: PreconvergenceConfig, enable_monitoring: bool = True):
        self.config = config
        self.stages = self._build_stages()
        self.checkpoint_manager = CheckpointManager(config.output_dir / "checkpoints")
        self.logger = StructuredLogger("Pipeline")
        
        # Inicializar monitor de producción si está habilitado
        self.monitor = None
        if enable_monitoring:
            # TODO: Implementar production_monitor cuando esté disponible
            # self.monitor = create_production_monitor(config)
            # self.logger.info("Production monitor initialized")
            self.logger.warning("Production monitor not available in this version")
        
        self.monitoring_enabled = enable_monitoring and self.monitor is not None

    def _build_stages(self) -> Dict[str, PipelineStage]:
        """Construye los stages del pipeline."""
        return {
            'cutoff': CutoffConvergenceStage(self.config),
            'kmesh': KMeshConvergenceStage(self.config),
            'lattice': LatticeOptimizationStage(self.config)
        }

    async def execute(self, resume_from: Optional[str] = None) -> PipelineResult:
        """Ejecuta el pipeline completo con manejo de checkpoints y monitoreo."""
        start_time = time.perf_counter()

        # Iniciar monitoreo si está habilitado
        if self.monitoring_enabled and self.monitor:
            self.monitor.start_monitoring(interval=1.0)
            self.logger.info("Production monitoring started")

        # Cargar estado si se reanuda
        if resume_from:
            state = self.checkpoint_manager.load_checkpoint(resume_from)
            completed_stages = state.get('completed_stages', [])
        else:
            completed_stages = []

        results = {}

        try:
            for stage_name, stage in self.stages.items():
                if stage_name in completed_stages:
                    self.logger.info(f"Skipping completed stage: {stage_name}")
                    results[stage_name] = self.checkpoint_manager.load_stage_result(stage_name)
                    continue

                self.logger.info(f"Executing stage: {stage_name}")

                # Ejecutar stage con monitoreo si está habilitado
                if self.monitoring_enabled and self.monitor:
                    # Usar contexto de monitoreo para el stage
                    stage_context_manager = self.monitor.stage_context(
                        stage_name,
                        stage_type=self._get_stage_type(stage_name)
                    )
                    
                    async with stage_context_manager:
                        try:
                            stage_result = await asyncio.wait_for(
                                stage.execute(results),
                                timeout=self.config.stage_timeout
                            )
                        except Exception as e:
                            # El context manager manejará el logging del error
                            raise
                else:
                    # Ejecución sin monitoreo
                    try:
                        stage_result = await asyncio.wait_for(
                            stage.execute(results),
                            timeout=self.config.stage_timeout
                        )
                    except Exception as e:
                        self.logger.error(f"Error in stage {stage_name}: {e}")
                        raise

                results[stage_name] = stage_result

                # Guardar checkpoint
                self.checkpoint_manager.save_stage_result(stage_name, stage_result)

                # Actualizar progreso
                completed_stages.append(stage_name)
                self.checkpoint_manager.save_progress(completed_stages)

                # Verificar si el stage falló
                if not stage_result.success:
                    raise RuntimeError(f"Stage {stage_name} failed: {stage_result.data}")

            total_duration = time.perf_counter() - start_time

            # Detener monitoreo y exportar métricas
            if self.monitoring_enabled and self.monitor:
                self.monitor.stop_monitoring()
                
                # Exportar métricas del pipeline
                metrics_file = self.config.output_dir / "monitoring" / f"pipeline_metrics_{int(time.time())}.json"
                self.monitor.export_metrics(metrics_file)
                
                self.logger.info(f"Pipeline completed successfully in {total_duration:.2f}s")
                self.logger.info(f"Monitoring data exported to: {metrics_file}")

            return PipelineResult(
                results=results,
                config=self.config,
                total_duration=total_duration,
                success=True
            )

        except Exception as e:
            total_duration = time.perf_counter() - start_time
            self.logger.error(f"Pipeline execution failed: {e}")

            # Detener monitoreo en caso de error
            if self.monitoring_enabled and self.monitor:
                self.monitor.stop_monitoring()
                
                # Exportar métricas de error
                error_metrics_file = self.config.output_dir / "monitoring" / f"pipeline_error_{int(time.time())}.json"
                self.monitor.export_metrics(error_metrics_file)
                
                self.logger.info(f"Error metrics exported to: {error_metrics_file}")

            return PipelineResult(
                results=results,
                config=self.config,
                total_duration=total_duration,
                success=False,
                error_message=str(e)
            )

    def _get_stage_type(self, stage_name: str) -> str:
        """Determina el tipo de stage para el monitoreo."""
        stage_types = {
            'cutoff': 'convergence_analysis',
            'kmesh': 'convergence_analysis',
            'lattice': 'optimization',
            'bands': 'electronic_structure',
            'slab': 'surface_analysis'
        }
        return stage_types.get(stage_name, 'unknown')

    async def _attempt_recovery(self, stage_name: str, error: Exception) -> bool:
        """Intenta recuperación automática de errores."""
        # Implementar lógica de recuperación
        # Por ahora, solo loggear
        self.logger.warning(f"Recovery attempted for stage {stage_name}: {error}")
        return False

    def get_stage_status(self, stage_name: str) -> Optional[StageResult]:
        """Obtiene estado de un stage específico."""
        return self.checkpoint_manager.load_stage_result(stage_name)

    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Obtiene progreso general del pipeline con información de monitoreo."""
        try:
            progress = self.checkpoint_manager.load_progress()
            completed_stages = progress.get('completed_stages', [])
            total_stages = len(self.stages)

            base_progress = {
                'completed_stages': completed_stages,
                'total_stages': total_stages,
                'progress_percentage': len(completed_stages) / total_stages * 100,
                'remaining_stages': [s for s in self.stages.keys() if s not in completed_stages]
            }
            
            # Agregar información de monitoreo si está disponible
            if self.monitoring_enabled and self.monitor:
                health_status = self.monitor.get_health_status()
                base_progress['monitoring'] = {
                    'enabled': True,
                    'health_status': health_status,
                    'current_metrics': health_status.get('current_metrics', {}),
                    'summary': health_status.get('summary', {})
                }
            else:
                base_progress['monitoring'] = {'enabled': False}
            
            return base_progress
        except Exception as e:
            return {
                'completed_stages': [],
                'total_stages': len(self.stages),
                'progress_percentage': 0,
                'remaining_stages': list(self.stages.keys()),
                'monitoring': {'enabled': self.monitoring_enabled},
                'error': str(e)
            }

    def get_monitoring_status(self) -> Optional[Dict[str, Any]]:
        """Obtiene estado detallado del monitoreo."""
        if not self.monitoring_enabled or not self.monitor:
            return None
        
        try:
            current_metrics = self.monitor.get_current_metrics()
            performance_summary = self.monitor.get_performance_summary()
            health_status = self.monitor.get_health_status()
            
            return {
                'monitoring_active': self.monitor.is_monitoring,
                'current_metrics': vars(current_metrics) if current_metrics else None,
                'performance_summary': performance_summary,
                'health_status': health_status,
                'recent_alerts': self.monitor.alerts[-5:] if self.monitor.alerts else [],
                'stage_history': self.monitor.stage_history[-5:] if self.monitor.stage_history else []
            }
        except Exception as e:
            return {
                'error': f"Failed to get monitoring status: {e}",
                'monitoring_active': self.monitor.is_monitoring if self.monitor else False
            }

    def export_monitoring_data(self, filepath: Path = None) -> bool:
        """Exporta datos de monitoreo a archivo."""
        if not self.monitoring_enabled or not self.monitor:
            return False
        
        if filepath is None:
            timestamp = int(time.time())
            filepath = self.config.output_dir / "monitoring" / f"export_{timestamp}.json"
        
        return self.monitor.export_metrics(filepath)
    
    def stop_monitoring(self):
        """Detiene el monitoreo manualmente."""
        if self.monitor and self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
            self.logger.info("Monitoring stopped manually")
    
    def get_system_requirements_check(self) -> Dict[str, Any]:
        """Verifica requisitos del sistema para el pipeline."""
        import psutil
        
        # Verificar memoria disponible
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calcular memoria requerida (estimación conservadora)
        max_kmesh = max(self.config.kmesh_list)
        nkpts = max_kmesh[0] * max_kmesh[1] * max_kmesh[2]
        estimated_memory_gb = 200 + nkpts * 50 / 1024  # MB a GB con factor de seguridad
        
        # Verificar CPU
        cpu_count = psutil.cpu_count()
        available_cores = max(1, cpu_count // 2)  # Usar la mitad como regla general
        
        # Verificar espacio en disco
        disk_usage = psutil.disk_usage('.').free / (1024**3)  # GB libres
        
        requirements_met = {
            'memory': {
                'available_gb': available_memory_gb,
                'total_gb': total_memory_gb,
                'required_gb': estimated_memory_gb,
                'meets_requirement': available_memory_gb >= estimated_memory_gb,
                'utilization_percent': (total_memory_gb - available_memory_gb) / total_memory_gb * 100
            },
            'cpu': {
                'total_cores': cpu_count,
                'recommended_cores': min(self.config.max_workers, available_cores),
                'meets_requirement': available_cores >= min(self.config.max_workers, 2)
            },
            'storage': {
                'free_gb': disk_usage,
                'required_gb': 10.0,  # Requisito mínimo conservador
                'meets_requirement': disk_usage >= 10.0
            }
        }
        
        overall_status = (
            requirements_met['memory']['meets_requirement'] and
            requirements_met['cpu']['meets_requirement'] and
            requirements_met['storage']['meets_requirement']
        )
        
        return {
            'overall_status': 'ready' if overall_status else 'inadequate',
            'requirements': requirements_met,
            'recommendations': self._generate_system_recommendations(requirements_met)
        }
    
    def _generate_system_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en los requisitos del sistema."""
        recommendations = []
        
        # Recomendaciones de memoria
        if not requirements['memory']['meets_requirement']:
            shortfall = requirements['memory']['required_gb'] - requirements['memory']['available_gb']
            recommendations.append(f"Consider reducing max_workers or using a system with {shortfall:.1f}GB more RAM")
        
        if requirements['memory']['utilization_percent'] > 80:
            recommendations.append("High memory usage detected - close other applications")
        
        # Recomendaciones de CPU
        if not requirements['cpu']['meets_requirement']:
            recommendations.append(f"Consider increasing max_workers from {self.config.max_workers} to {requirements['cpu']['recommended_cores']}")
        
        # Recomendaciones de almacenamiento
        if not requirements['storage']['meets_requirement']:
            recommendations.append("Insufficient disk space - free up at least 10GB for safe operation")
        
        return recommendations


# Funciones de utilidad para integración
async def run_preconvergence_pipeline(config: PreconvergenceConfig,
                                    resume_from: Optional[str] = None) -> PipelineResult:
    """Función de alto nivel para ejecutar el pipeline completo."""
    pipeline = PreconvergencePipeline(config)
    return await pipeline.execute(resume_from)


def create_pipeline_from_config(config_path: Path) -> PreconvergencePipeline:
    """Crea pipeline desde archivo de configuración."""
    from ..config.settings import PreconvergenceConfig

    config = PreconvergenceConfig.load_from_file(config_path)
    return PreconvergencePipeline(config)


# === FUNCIONES DE INTEGRACIÓN MULTIMATERIAL ===

def run_single_material_pipeline(config: PreconvergenceConfig,
                                material_name: str = "Material") -> PipelineResult:
    """
    Ejecuta pipeline para un solo material (compatibilidad con multimaterial).
    
    Args:
        config: Configuración del pipeline
        material_name: Nombre del material para logging
        
    Returns:
        PipelineResult con el resultado
    """
    pipeline = PreconvergencePipeline(config)
    return asyncio.run(pipeline.execute())


def create_multi_material_pipeline_from_config(config_path: Path) -> Optional[MultiMaterialPipeline]:
    """
    Crea pipeline multimaterial desde archivo de configuración.
    
    Args:
        config_path: Ruta al archivo de configuración YAML
        
    Returns:
        MultiMaterialPipeline configurado o None si no está disponible
    """
    if not MULTIMATERIAL_AVAILABLE:
        return None
    
    try:
        from ..core.multi_material_config import MultiMaterialConfig
        multi_config = MultiMaterialConfig.load_from_file(config_path)
        return MultiMaterialPipeline(multi_config)
    except Exception as e:
        print(f"Error creando pipeline multimaterial: {e}")
        return None


# Funciones de conveniencia para migración
def migrate_to_multi_material(single_config: PreconvergenceConfig,
                            materials: List[str]) -> Optional[MultiMaterialPipeline]:
    """
    Migra configuración de material único a multimaterial.
    
    Args:
        single_config: Configuración de material único
        materials: Lista de materiales a procesar
        
    Returns:
        MultiMaterialPipeline configurado o None si no está disponible
    """
    if not MULTIMATERIAL_AVAILABLE:
        return None
    
    try:
        from ..core.multi_material_config import MultiMaterialConfig
        
        # Crear configuración multimaterial basada en la individual
        multi_config = MultiMaterialConfig(
            base_config=single_config,
            output_base_dir=Path("results_migrated")
        )
        
        # Agregar materiales
        for material in materials:
            multi_config.add_material(material)
        
        return MultiMaterialPipeline(multi_config)
        
    except Exception as e:
        print(f"Error migrando a configuración multimaterial: {e}")
        return None


def is_multi_material_available() -> bool:
    """Verifica si el sistema multimaterial está disponible."""
    return MULTIMATERIAL_AVAILABLE


# Función de compatibilidad para ejecutar campañas
async def run_campaign_compatibility(materials: List[str],
                                   config: Optional[PreconvergenceConfig] = None,
                                   parallel: bool = True,
                                   max_workers: int = 4) -> Optional[CampaignResult]:
    """
    Función de compatibilidad para ejecutar campañas multimaterial.
    
    Args:
        materials: Lista de materiales a procesar
        config: Configuración base (usa default si None)
        parallel: Ejecutar en paralelo
        max_workers: Número máximo de workers
        
    Returns:
        CampaignResult o None si no está disponible
    """
    if not MULTIMATERIAL_AVAILABLE:
        print("Sistema multimaterial no disponible. Use run_single_material_pipeline para materiales individuales.")
        return None
    
    try:
        return await run_preconvergence_campaign(
            materials=materials,
            parallel=parallel,
            max_workers=max_workers
        )
    except Exception as e:
        print(f"Error ejecutando campaña: {e}")
        return None


# Función para mostrar capacidades multimaterial
def show_multi_material_capabilities():
    """Muestra las capacidades del sistema multimaterial."""
    print("=== CAPACIDADES MULTIMATERIAL ===")
    print(f"Sistema disponible: {MULTIMATERIAL_AVAILABLE}")
    
    if MULTIMATERIAL_AVAILABLE:
        print("✅ Funcionalidades disponibles:")
        print("  • Ejecución de múltiples materiales")
        print("  • Paralelización automática")
        print("  • Análisis comparativo")
        print("  • Gestión de memoria inteligente")
        print("  • Checkpoints por material")
        print("  • Reportes de progreso")
        print("  • Análisis estadístico avanzado")
        print("  • Visualizaciones automáticas")
    else:
        print("❌ Funcionalidades multimaterial no disponibles")
        print("  • Solo ejecución de material único")
        print("  • Pipeline básico de preconvergencia")


# Función para validar configuración antes de ejecutar
def validate_multi_material_setup() -> Dict[str, Any]:
    """
    Valida el setup del sistema multimaterial.
    
    Returns:
        Dict con información de validación
    """
    validation = {
        'multi_material_available': MULTIMATERIAL_AVAILABLE,
        'dependencies_ok': True,
        'warnings': [],
        'errors': []
    }
    
    # Verificar dependencias
    try:
        import asyncio
        import concurrent.futures
        import numpy as np
        import pandas as pd
    except ImportError as e:
        validation['dependencies_ok'] = False
        validation['errors'].append(f"Dependencia faltante: {e}")
    
    # Verificar configuración de memoria
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            validation['warnings'].append(f"Poca memoria disponible: {memory_gb:.1f}GB")
    except ImportError:
        validation['warnings'].append("psutil no disponible - no se puede verificar memoria")
    
    # Verificar espacio en disco
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    if free_space_gb < 10:
        validation['warnings'].append(f"Poco espacio en disco: {free_space_gb:.1f}GB libres")
    
    return validation