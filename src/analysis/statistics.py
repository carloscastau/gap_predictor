# src/analysis/statistics.py
"""Análisis estadístico de resultados de convergencia."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.logging import StructuredLogger


class ConvergenceStatistics:
    """Estadísticas de convergencia de parámetros DFT."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger("ConvergenceStats")

    def analyze_cutoff_convergence(self, cutoff_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza estadísticas de convergencia vs cutoff."""
        if cutoff_data.empty or 'E_tot_Ha' not in cutoff_data.columns:
            return {'error': 'Datos de cutoff insuficientes'}

        energies = cutoff_data['E_tot_Ha'].values
        cutoffs = cutoff_data['ke_cutoff_Ry'].values

        # Energía mínima
        e_min = np.min(energies)
        e_min_idx = np.argmin(energies)

        # Calcular convergencia
        convergence_analysis = self._calculate_convergence_metrics(
            cutoffs, energies, e_min, "cutoff"
        )

        # Estadísticas adicionales
        stats = {
            'optimal_cutoff': cutoffs[e_min_idx],
            'min_energy': e_min,
            'energy_range': np.max(energies) - e_min,
            'std_deviation': np.std(energies),
            'coefficient_of_variation': np.std(energies) / abs(np.mean(energies)) if np.mean(energies) != 0 else np.nan,
            'n_points': len(energies),
            'convergence_analysis': convergence_analysis
        }

        self.logger.info(f"Cutoff analysis: optimal={stats['optimal_cutoff']} Ry, "
                        f"E_min={stats['min_energy']:.6f} Ha")

        return stats

    def analyze_kmesh_convergence(self, kmesh_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza estadísticas de convergencia vs k-mesh."""
        if kmesh_data.empty or 'E_tot_Ha' not in kmesh_data.columns:
            return {'error': 'Datos de k-mesh insuficientes'}

        energies = kmesh_data['E_tot_Ha'].values
        nkpts = kmesh_data['N_kpts'].values

        # Energía mínima
        e_min = np.min(energies)
        e_min_idx = np.argmin(energies)

        # Calcular convergencia
        convergence_analysis = self._calculate_convergence_metrics(
            nkpts, energies, e_min, "k-mesh"
        )

        # Estadísticas adicionales
        stats = {
            'optimal_nkpts': nkpts[e_min_idx],
            'min_energy': e_min,
            'energy_range': np.max(energies) - e_min,
            'std_deviation': np.std(energies),
            'coefficient_of_variation': np.std(energies) / abs(np.mean(energies)) if np.mean(energies) != 0 else np.nan,
            'n_points': len(energies),
            'convergence_analysis': convergence_analysis
        }

        self.logger.info(f"k-mesh analysis: optimal={stats['optimal_nkpts']} k-points, "
                        f"E_min={stats['min_energy']:.6f} Ha")

        return stats

    def analyze_lattice_optimization(self, lattice_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza estadísticas de optimización de parámetro de red."""
        if lattice_data.empty or 'E_tot_Ha' not in lattice_data.columns:
            return {'error': 'Datos de lattice insuficientes'}

        energies = lattice_data['E_tot_Ha'].values
        lattice_constants = lattice_data['a_Ang'].values

        # Energía mínima
        e_min = np.min(energies)
        e_min_idx = np.argmin(energies)

        # Ajuste parabólico si hay suficientes puntos
        fit_results = {}
        if len(lattice_constants) >= 3:
            fit_results = self._fit_parabolic_potential(lattice_constants, energies)

        stats = {
            'optimal_lattice': lattice_constants[e_min_idx],
            'min_energy': e_min,
            'energy_range': np.max(energies) - e_min,
            'lattice_range': np.max(lattice_constants) - np.min(lattice_constants),
            'n_points': len(energies),
            'parabolic_fit': fit_results
        }

        self.logger.info(f"Lattice analysis: optimal={stats['optimal_lattice']:.4f} Å, "
                        f"E_min={stats['min_energy']:.6f} Ha")

        return stats

    def _calculate_convergence_metrics(self, parameters: np.ndarray, energies: np.ndarray,
                                     e_min: float, param_name: str) -> Dict[str, Any]:
        """Calcula métricas de convergencia."""
        # Ordenar por parámetro
        sort_idx = np.argsort(parameters)
        params_sorted = parameters[sort_idx]
        energies_sorted = energies[sort_idx]

        # Calcular diferencias de energía
        energy_diffs = np.abs(energies_sorted - e_min)

        # Encontrar punto de convergencia (donde ΔE < threshold)
        threshold = 1e-4  # Ha
        converged_idx = None

        for i in range(len(energy_diffs)):
            if energy_diffs[i] <= threshold:
                # Verificar que los siguientes puntos también converjan
                remaining_diffs = energy_diffs[i:]
                if len(remaining_diffs) >= 2 and np.max(remaining_diffs) <= threshold:
                    converged_idx = i
                    break

        convergence_point = params_sorted[converged_idx] if converged_idx is not None else None

        # Calcular ratio de convergencia
        if len(energies_sorted) > 1:
            convergence_ratio = 1.0 - (energy_diffs[-1] / energy_diffs[0]) if energy_diffs[0] != 0 else 0.0
        else:
            convergence_ratio = 0.0

        return {
            'converged': converged_idx is not None,
            'convergence_point': convergence_point,
            'convergence_threshold': threshold,
            'convergence_ratio': convergence_ratio,
            'max_energy_diff': np.max(energy_diffs),
            'mean_energy_diff': np.mean(energy_diffs)
        }

    def _fit_parabolic_potential(self, lattice_constants: np.ndarray,
                               energies: np.ndarray) -> Dict[str, Any]:
        """Ajusta potencial parabólico E(a) = A*(a-a0)² + E0."""
        try:
            # Centrar en el mínimo observado
            a_min_idx = np.argmin(energies)
            a0 = lattice_constants[a_min_idx]
            E0 = energies[a_min_idx]

            # Variables centradas
            a_centered = lattice_constants - a0
            E_centered = energies - E0

            # Ajuste lineal: E = A * a²
            coeffs = np.polyfit(a_centered**2, E_centered, 1)
            A = coeffs[0]

            # Calcular a_opt teórica (debería ser a0)
            a_opt_theory = a0

            # Calidad del ajuste
            E_fit = A * a_centered**2 + E0
            ss_res = np.sum((energies - E_fit)**2)
            ss_tot = np.sum((energies - np.mean(energies))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            # Frecuencia de vibración (aproximada)
            # E = (1/2) * k * x², donde k = 2*A (en unidades atómicas)
            # frecuencia = sqrt(k/m) / 2π, pero necesitamos masa reducida
            # Aquí solo reportamos la constante de fuerza
            force_constant = 2 * A  # Ha/Å²

            return {
                'a0': a0,
                'E0': E0,
                'A': A,
                'force_constant': force_constant,
                'r2': r2,
                'fit_quality': 'excellent' if r2 > 0.95 else 'good' if r2 > 0.8 else 'poor'
            }

        except Exception as e:
            return {'error': f'Parabolic fit failed: {str(e)}'}


class PerformanceAnalyzer:
    """Analizador de rendimiento del pipeline."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger("PerformanceAnalyzer")

    def analyze_stage_performance(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza rendimiento de stages."""
        performance_stats = {}

        for stage_name, result in stage_results.items():
            if hasattr(result, 'duration'):
                duration = result.duration
            elif isinstance(result, dict) and 'duration' in result:
                duration = result['duration']
            else:
                continue

            performance_stats[stage_name] = {
                'duration': duration,
                'efficiency': self._calculate_stage_efficiency(stage_name, result)
            }

        # Estadísticas globales
        total_time = sum(stats['duration'] for stats in performance_stats.values())
        slowest_stage = max(performance_stats.items(), key=lambda x: x[1]['duration'])

        performance_stats['summary'] = {
            'total_time': total_time,
            'n_stages': len(performance_stats),
            'slowest_stage': slowest_stage[0],
            'slowest_duration': slowest_stage[1]['duration'],
            'average_stage_time': total_time / len(performance_stats) if performance_stats else 0
        }

        self.logger.info(f"Performance analysis: total_time={total_time:.2f}s, "
                        f"slowest={slowest_stage[0]} ({slowest_stage[1]['duration']:.2f}s)")

        return performance_stats

    def _calculate_stage_efficiency(self, stage_name: str, result: Any) -> float:
        """Calcula eficiencia de un stage (operaciones por segundo)."""
        # Implementación simplificada
        # En la práctica, contaría cálculos DFT realizados
        return 1.0  # Placeholder

    def generate_performance_report(self, performance_stats: Dict[str, Any],
                                  output_path: Path) -> None:
        """Genera reporte de rendimiento."""
        report = f"""
PERFORMANCE ANALYSIS REPORT
{'='*50}

Total execution time: {performance_stats.get('summary', {}).get('total_time', 0):.2f}s
Number of stages: {performance_stats.get('summary', {}).get('n_stages', 0)}
Average stage time: {performance_stats.get('summary', {}).get('average_stage_time', 0):.2f}s

Stage breakdown:
"""

        for stage_name, stats in performance_stats.items():
            if stage_name != 'summary':
                report += f"  {stage_name}: {stats['duration']:.2f}s\n"

        with open(output_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Performance report saved to {output_path}")