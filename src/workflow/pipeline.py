# src/workflow/pipeline.py
"""Pipeline principal con stages modulares e independientes."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from ..config.settings import PreconvergenceConfig
    from ..core.calculator import DFTCalculator, CellParameters
    from ..core.optimizer import LatticeOptimizer, ConvergenceAnalyzer
    from ..core.parallel import TaskScheduler, CalculationTask
    from ..workflow.checkpoint import CheckpointManager
    from ..utils.logging import StructuredLogger
except ImportError:
    # Fallback para imports absolutos cuando se ejecuta como script
    from config.settings import PreconvergenceConfig
    from core.calculator import DFTCalculator, CellParameters
    from core.optimizer import LatticeOptimizer, ConvergenceAnalyzer
    from core.parallel import TaskScheduler, CalculationTask
    from workflow.checkpoint import CheckpointManager
    from utils.logging import StructuredLogger


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
                x_ga=self.config.x_g,
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

    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self.stages = self._build_stages()
        self.checkpoint_manager = CheckpointManager(config.output_dir / "checkpoints")
        self.logger = StructuredLogger("Pipeline")

    def _build_stages(self) -> Dict[str, PipelineStage]:
        """Construye los stages del pipeline."""
        return {
            'cutoff': CutoffConvergenceStage(self.config),
            'kmesh': KMeshConvergenceStage(self.config),
            'lattice': LatticeOptimizationStage(self.config)
        }

    async def execute(self, resume_from: Optional[str] = None) -> PipelineResult:
        """Ejecuta el pipeline completo con manejo de checkpoints."""
        start_time = time.perf_counter()

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

                # Ejecutar stage con timeout y manejo de errores
                try:
                    stage_result = await asyncio.wait_for(
                        stage.execute(results),
                        timeout=self.config.stage_timeout
                    )

                    results[stage_name] = stage_result

                    # Guardar checkpoint
                    self.checkpoint_manager.save_stage_result(stage_name, stage_result)

                    # Actualizar progreso
                    completed_stages.append(stage_name)
                    self.checkpoint_manager.save_progress(completed_stages)

                    # Verificar si el stage falló
                    if not stage_result.success:
                        raise RuntimeError(f"Stage {stage_name} failed: {stage_result.data}")

                except asyncio.TimeoutError:
                    self.logger.error(f"Stage {stage_name} exceeded timeout")
                    raise
                except Exception as e:
                    self.logger.error(f"Error in stage {stage_name}: {e}")
                    # Intentar recuperación automática
                    if await self._attempt_recovery(stage_name, e):
                        continue
                    raise

            total_duration = time.perf_counter() - start_time

            return PipelineResult(
                results=results,
                config=self.config,
                total_duration=total_duration,
                success=True
            )

        except Exception as e:
            total_duration = time.perf_counter() - start_time
            self.logger.error(f"Pipeline execution failed: {e}")

            return PipelineResult(
                results=results,
                config=self.config,
                total_duration=total_duration,
                success=False,
                error_message=str(e)
            )

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
        """Obtiene progreso general del pipeline."""
        try:
            progress = self.checkpoint_manager.load_progress()
            completed_stages = progress.get('completed_stages', [])
            total_stages = len(self.stages)

            return {
                'completed_stages': completed_stages,
                'total_stages': total_stages,
                'progress_percentage': len(completed_stages) / total_stages * 100,
                'remaining_stages': [s for s in self.stages.keys() if s not in completed_stages]
            }
        except:
            return {
                'completed_stages': [],
                'total_stages': len(self.stages),
                'progress_percentage': 0,
                'remaining_stages': list(self.stages.keys())
            }


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