# src/workflow/stages/base.py
"""Clases base para stages del pipeline."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ...config.settings import PreconvergenceConfig
from ...utils.logging import StructuredLogger


@dataclass
class StageResult:
    """Resultado estándar de un stage."""
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    duration: float
    timestamp: str
    stage_name: str


class PipelineStage(ABC):
    """Interface común para todos los stages del pipeline."""

    def __init__(self, config: PreconvergenceConfig, name: str):
        self.config = config
        self.name = name
        self.logger = StructuredLogger(f"Stage.{name}")

    @abstractmethod
    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta el stage y retorna resultado."""
        pass

    @abstractmethod
    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        """Valida que las entradas del stage sean correctas."""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Retorna lista de stages de los que depende este stage."""
        pass

    async def _execute_with_timing(self, operation) -> tuple:
        """Ejecuta operación con medición de tiempo."""
        import time
        start_time = time.perf_counter()
        result = await operation()
        duration = time.perf_counter() - start_time
        return result, duration

    def _collect_metrics(self) -> Dict[str, Any]:
        """Recopila métricas del stage."""
        import psutil
        import os

        try:
            process = psutil.Process(os.getpid())
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except:
            return {}

    def _create_result(self, success: bool, data: Dict[str, Any],
                      duration: float, error: Optional[str] = None) -> StageResult:
        """Crea resultado estándar del stage."""
        import time

        result_data = data.copy()
        if error:
            result_data['error'] = error

        return StageResult(
            success=success,
            data=result_data,
            metrics=self._collect_metrics(),
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            stage_name=self.name
        )


class ConvergenceStage(PipelineStage):
    """Stage base para análisis de convergencia."""

    def __init__(self, config: PreconvergenceConfig, name: str, parameter_name: str):
        super().__init__(config, name)
        self.parameter_name = parameter_name

    async def execute_convergence_analysis(self, parameter_values: List[float],
                                         energy_calculator, threshold: float) -> Dict[str, Any]:
        """Ejecuta análisis de convergencia genérico."""
        from ...core.optimizer import ConvergencePoint, ConvergenceAnalyzer

        points = []
        for param_value in parameter_values:
            try:
                energy = await energy_calculator(param_value)
                if not (isinstance(energy, float) and not np.isnan(energy)):
                    self.logger.warning(f"Invalid energy for {self.parameter_name}={param_value}")
                    continue

                points.append(ConvergencePoint(
                    parameter=param_value,
                    energy=energy
                ))

            except Exception as e:
                self.logger.error(f"Failed to calculate energy for {self.parameter_name}={param_value}: {e}")
                continue

        if len(points) < 3:
            return {
                'error': f'Insufficient valid points for {self.parameter_name} convergence analysis',
                'points_collected': len(points)
            }

        # Analizar convergencia
        analyzer = ConvergenceAnalyzer(self.config)

        if self.parameter_name == 'cutoff':
            result = analyzer.analyze_cutoff_convergence(points, threshold)
        elif self.parameter_name == 'kmesh':
            result = analyzer.analyze_kmesh_convergence(points, threshold)
        else:
            # Análisis genérico
            result = analyzer.analyze_cutoff_convergence(points, threshold)

        return {
            'parameter_values': [p.parameter for p in points],
            'energies': [p.energy for p in points],
            'optimal_value': result.optimal_value,
            'converged': result.converged,
            'threshold_used': result.convergence_threshold,
            'points_analyzed': result.points_used,
            'fit_quality': result.fit_quality
        }


class OptimizationStage(PipelineStage):
    """Stage base para optimizaciones."""

    def __init__(self, config: PreconvergenceConfig, name: str):
        super().__init__(config, name)

    async def execute_optimization(self, optimizer_func, bounds: tuple,
                                 initial_guess: Optional[float] = None) -> Dict[str, Any]:
        """Ejecuta optimización genérica."""
        from ...core.optimizer import LatticeOptimizer

        try:
            optimizer = LatticeOptimizer(self.config)

            # Usar optimización avanzada si está disponible
            if hasattr(optimizer, 'optimize_lattice_constant'):
                result = await optimizer.optimize_lattice_constant(
                    optimizer_func,
                    a_range=bounds,
                    n_points=7
                )
            else:
                # Fallback a optimización simple
                result = await self._simple_optimization(optimizer_func, bounds, initial_guess)

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return {'error': str(e)}

    async def _simple_optimization(self, func, bounds: tuple, initial_guess: Optional[float]) -> Dict[str, Any]:
        """Optimización simple como fallback."""
        import time

        start_time = time.perf_counter()

        # Evaluación en puntos de la grilla
        a_min, a_max = bounds
        points = []
        n_points = 7

        for i in range(n_points):
            a = a_min + (a_max - a_min) * i / (n_points - 1)
            try:
                energy = await func(a)
                if isinstance(energy, float) and not np.isnan(energy):
                    points.append((a, energy))
            except:
                continue

        if not points:
            return {'error': 'No valid points found during optimization'}

        # Encontrar mínimo
        points.sort(key=lambda x: x[1])
        a_opt, e_min = points[0]

        computation_time = time.perf_counter() - start_time

        return {
            'a_opt': a_opt,
            'e_min': e_min,
            'all_points': points,
            'n_total_points': len(points),
            'computation_time': computation_time,
            'success': True,
            'warning': 'Simple optimization used (advanced optimization not available)'
        }


# Funciones de utilidad
def create_stage_result(success: bool, data: Dict[str, Any], stage_name: str,
                       duration: float) -> StageResult:
    """Crea resultado de stage de forma conveniente."""
    import time

    return StageResult(
        success=success,
        data=data,
        metrics={},
        duration=duration,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        stage_name=stage_name
    )


def validate_stage_dependencies(stage: PipelineStage,
                               previous_results: Dict[str, StageResult]) -> bool:
    """Valida dependencias de un stage."""
    dependencies = stage.get_dependencies()

    for dep in dependencies:
        if dep not in previous_results:
            return False

        # Verificar que el stage dependiente fue exitoso
        if not previous_results[dep].success:
            return False

    return True