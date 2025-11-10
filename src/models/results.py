# src/models/results.py
"""Modelos de datos para resultados de cálculos."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class CalculationResult:
    """Resultado de un cálculo individual."""

    task_id: str
    success: bool
    energy: float
    converged: bool
    n_iterations: Optional[int] = None
    computation_time: float = 0.0
    memory_peak: float = 0.0
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def energy_ev(self) -> float:
        """Energía en eV."""
        return self.energy * 27.211386245988 if np.isfinite(self.energy) else np.nan

    @property
    def is_valid(self) -> bool:
        """Verifica si el resultado es válido."""
        return self.success and self.converged and np.isfinite(self.energy)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'energy': self.energy,
            'energy_ev': self.energy_ev,
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'computation_time': self.computation_time,
            'memory_peak': self.memory_peak,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid
        }


@dataclass
class ConvergenceResult:
    """Resultado de análisis de convergencia."""

    parameter_name: str
    converged: bool
    optimal_value: Optional[float] = None
    convergence_threshold: float = 1e-4
    points_analyzed: int = 0
    fit_quality: float = 0.0
    confidence_interval: Optional[Tuple[float, float]] = None
    parameter_values: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)

    @property
    def convergence_ratio(self) -> float:
        """Ratio de convergencia (0-1)."""
        if not self.energies:
            return 0.0

        energies = np.array(self.energies)
        if len(energies) < 2:
            return 0.0

        # Calcular cambio relativo máximo
        emin = np.min(energies)
        max_change = np.max(np.abs(energies - emin))
        return 1.0 - min(1.0, max_change / abs(emin)) if emin != 0 else 0.0

    @property
    def has_sufficient_points(self) -> bool:
        """Verifica si hay suficientes puntos para análisis."""
        return len(self.parameter_values) >= 3

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'parameter_name': self.parameter_name,
            'converged': self.converged,
            'optimal_value': self.optimal_value,
            'convergence_threshold': self.convergence_threshold,
            'points_analyzed': self.points_analyzed,
            'fit_quality': self.fit_quality,
            'confidence_interval': self.confidence_interval,
            'parameter_values': self.parameter_values,
            'energies': self.energies,
            'convergence_ratio': self.convergence_ratio,
            'has_sufficient_points': self.has_sufficient_points
        }


@dataclass
class OptimizationResult:
    """Resultado de optimización de parámetros."""

    parameter_name: str
    optimal_value: float
    optimal_energy: float
    uncertainty: Optional[float] = None
    fit_quality: float = 0.0
    points_evaluated: int = 0
    computation_time: float = 0.0
    method_used: str = "unknown"
    all_points: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def energy_ev(self) -> float:
        """Energía óptima en eV."""
        return self.optimal_energy * 27.211386245988 if np.isfinite(self.optimal_energy) else np.nan

    @property
    def relative_uncertainty(self) -> Optional[float]:
        """Incertidumbre relativa."""
        if self.uncertainty is None or self.optimal_value == 0:
            return None
        return abs(self.uncertainty / self.optimal_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'parameter_name': self.parameter_name,
            'optimal_value': self.optimal_value,
            'optimal_energy': self.optimal_energy,
            'optimal_energy_ev': self.energy_ev,
            'uncertainty': self.uncertainty,
            'relative_uncertainty': self.relative_uncertainty,
            'fit_quality': self.fit_quality,
            'points_evaluated': self.points_evaluated,
            'computation_time': self.computation_time,
            'method_used': self.method_used,
            'all_points': self.all_points
        }


@dataclass
class BandStructureResult:
    """Resultado de cálculo de estructura de bandas."""

    kpoints: np.ndarray
    bands: np.ndarray
    fermi_level: Optional[float] = None
    gap: Optional[float] = None
    gap_type: Optional[str] = None  # 'direct' or 'indirect'
    vbm_kpoint: Optional[np.ndarray] = None
    cbm_kpoint: Optional[np.ndarray] = None
    computation_time: float = 0.0

    @property
    def n_bands(self) -> int:
        """Número de bandas."""
        return self.bands.shape[1] if self.bands.ndim > 1 else 0

    @property
    def n_kpoints(self) -> int:
        """Número de puntos k."""
        return self.kpoints.shape[0] if self.kpoints.ndim > 1 else 0

    @property
    def gap_ev(self) -> Optional[float]:
        """Gap en eV."""
        return self.gap * 27.211386245988 if self.gap else None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'kpoints': self.kpoints.tolist() if self.kpoints is not None else None,
            'bands': self.bands.tolist() if self.bands is not None else None,
            'fermi_level': self.fermi_level,
            'gap': self.gap,
            'gap_ev': self.gap_ev,
            'gap_type': self.gap_type,
            'vbm_kpoint': self.vbm_kpoint.tolist() if self.vbm_kpoint is not None else None,
            'cbm_kpoint': self.cbm_kpoint.tolist() if self.cbm_kpoint is not None else None,
            'computation_time': self.computation_time,
            'n_bands': self.n_bands,
            'n_kpoints': self.n_kpoints
        }


@dataclass
class DOSResult:
    """Resultado de cálculo de densidad de estados."""

    energies: np.ndarray
    dos: np.ndarray
    fermi_level: Optional[float] = None
    integrated_dos: Optional[np.ndarray] = None
    computation_time: float = 0.0

    @property
    def energy_range(self) -> Tuple[float, float]:
        """Rango de energías."""
        return (np.min(self.energies), np.max(self.energies))

    @property
    def total_states(self) -> Optional[float]:
        """Número total de estados."""
        return np.trapz(self.dos, self.energies) if self.integrated_dos is None else self.integrated_dos[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'energies': self.energies.tolist(),
            'dos': self.dos.tolist(),
            'fermi_level': self.fermi_level,
            'integrated_dos': self.integrated_dos.tolist() if self.integrated_dos is not None else None,
            'computation_time': self.computation_time,
            'energy_range': self.energy_range,
            'total_states': self.total_states
        }


@dataclass
class SlabResult:
    """Resultado de cálculo de superficie (slab)."""

    miller_indices: Tuple[int, int, int]
    surface_energy: float  # eV/Å²
    work_function: float   # eV
    electrostatic_potential: np.ndarray
    z_positions: np.ndarray
    vacuum_level: Optional[float] = None
    computation_time: float = 0.0

    @property
    def miller_string(self) -> str:
        """String de índices Miller."""
        return f"({self.miller_indices[0]}{self.miller_indices[1]}{self.miller_indices[2]})"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'miller_indices': self.miller_indices,
            'miller_string': self.miller_string,
            'surface_energy': self.surface_energy,
            'work_function': self.work_function,
            'electrostatic_potential': self.electrostatic_potential.tolist(),
            'z_positions': self.z_positions.tolist(),
            'vacuum_level': self.vacuum_level,
            'computation_time': self.computation_time
        }


# Funciones de utilidad
def create_calculation_result(task_id: str, energy: float, converged: bool = True,
                             n_iterations: Optional[int] = None,
                             computation_time: float = 0.0) -> CalculationResult:
    """Crea resultado de cálculo de forma conveniente."""
    return CalculationResult(
        task_id=task_id,
        success=converged,
        energy=energy,
        converged=converged,
        n_iterations=n_iterations,
        computation_time=computation_time
    )


def merge_convergence_results(*results: ConvergenceResult) -> ConvergenceResult:
    """Combina múltiples resultados de convergencia."""
    if not results:
        return ConvergenceResult("unknown", False)

    # Usar el primer resultado como base
    merged = ConvergenceResult(
        parameter_name=results[0].parameter_name,
        converged=all(r.converged for r in results),
        optimal_value=np.mean([r.optimal_value for r in results if r.optimal_value]),
        convergence_threshold=min(r.convergence_threshold for r in results),
        points_analyzed=sum(r.points_analyzed for r in results),
        fit_quality=np.mean([r.fit_quality for r in results])
    )

    return merged