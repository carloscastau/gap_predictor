# src/core/optimizer.py
"""Optimizadores para convergencia de parámetros DFT."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import asyncio
from scipy.optimize import minimize_scalar, curve_fit
import time

from ..config.settings import PreconvergenceConfig


@dataclass
class ConvergencePoint:
    """Punto de datos para análisis de convergencia."""
    parameter: float
    energy: float
    error: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConvergenceResult:
    """Resultado de análisis de convergencia."""
    converged: bool
    optimal_value: float
    convergence_threshold: float
    points_used: int
    fit_quality: float
    confidence_interval: Optional[Tuple[float, float]] = None


class ConvergenceAnalyzer:
    """Analizador de convergencia de parámetros."""

    def __init__(self, config: PreconvergenceConfig):
        self.config = config

    def analyze_cutoff_convergence(self, points: List[ConvergencePoint],
                                 threshold: float = 1e-4) -> ConvergenceResult:
        """Analiza convergencia vs cutoff."""
        if len(points) < 3:
            return ConvergenceResult(False, np.nan, threshold, len(points), 0.0)

        # Ordenar por cutoff
        sorted_points = sorted(points, key=lambda p: p.parameter)
        energies = np.array([p.energy for p in sorted_points])
        cutoffs = np.array([p.parameter for p in sorted_points])

        # Encontrar punto de convergencia
        emin = np.min(energies)
        for i, (cutoff, energy) in enumerate(zip(cutoffs, energies)):
            if abs(energy - emin) <= threshold:
                # Verificar que los siguientes puntos también estén cerca
                remaining = energies[i:]
                if len(remaining) >= 2 and np.max(remaining) - np.min(remaining) <= threshold:
                    return ConvergenceResult(
                        True, cutoff, threshold, len(points),
                        self._calculate_fit_quality(cutoffs, energies)
                    )

        return ConvergenceResult(False, cutoffs[-1], threshold, len(points), 0.0)

    def analyze_kmesh_convergence(self, points: List[ConvergencePoint],
                                threshold: float = 1e-5) -> ConvergenceResult:
        """Analiza convergencia vs k-mesh."""
        if len(points) < 3:
            return ConvergenceResult(False, np.nan, threshold, len(points), 0.0)

        # Convertir k-mesh a número total de puntos
        def kmesh_to_nkpts(kmesh_str: str) -> int:
            # Asumir formato "kx x ky x kz"
            parts = kmesh_str.replace('x', ' ').split()
            return int(parts[0]) * int(parts[1]) * int(parts[2])

        nkpts_list = []
        energies = []
        for point in points:
            try:
                nkpts = kmesh_to_nkpts(str(point.parameter))
                nkpts_list.append(nkpts)
                energies.append(point.energy)
            except:
                continue

        if len(nkpts_list) < 3:
            return ConvergenceResult(False, np.nan, threshold, len(points), 0.0)

        nkpts_array = np.array(nkpts_list)
        energies_array = np.array(energies)

        # Ordenar por nkpts
        sort_idx = np.argsort(nkpts_array)
        nkpts_sorted = nkpts_array[sort_idx]
        energies_sorted = energies_array[sort_idx]

        # Encontrar convergencia
        emin = np.min(energies_sorted)
        for i, (nkpts, energy) in enumerate(zip(nkpts_sorted, energies_sorted)):
            if abs(energy - emin) <= threshold:
                remaining = energies_sorted[i:]
                if len(remaining) >= 2 and np.max(remaining) - np.min(remaining) <= threshold:
                    return ConvergenceResult(
                        True, nkpts, threshold, len(points),
                        self._calculate_fit_quality(nkpts_sorted, energies_sorted)
                    )

        return ConvergenceResult(False, nkpts_sorted[-1], threshold, len(points), 0.0)

    def _calculate_fit_quality(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula calidad del ajuste (R²)."""
        try:
            coeffs = np.polyfit(x, y, 2, cov=True)
            y_fit = np.polyval(coeffs[0], x)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            return max(0, min(1, r2))  # Clamp to [0,1]
        except:
            return 0.0


class LatticeOptimizer:
    """Optimizador de parámetro de red."""

    def __init__(self, config: PreconvergenceConfig):
        self.config = config

    async def optimize_lattice_constant(self, energy_calculator: Callable,
                                      a_range: Tuple[float, float] = (5.4, 5.8),
                                      n_points: int = 7) -> Dict[str, Any]:
        """Optimiza parámetro de red usando búsqueda global + refinamiento local."""

        start_time = time.perf_counter()

        # Fase 1: Exploración amplia
        print("🔍 Fase 1: Exploración amplia del espacio de parámetros")
        exploration_points = np.linspace(a_range[0], a_range[1], n_points)

        exploration_results = []
        for a in exploration_points:
            try:
                energy = await energy_calculator(a)
                exploration_results.append((a, energy))
                print(".4f"            except Exception as e:
                print(".4f"                exploration_results.append((a, np.nan))

        # Filtrar puntos válidos
        valid_results = [(a, e) for a, e in exploration_results if np.isfinite(e)]
        if len(valid_results) < 3:
            return {'error': 'Insuficientes puntos válidos para optimización'}

        # Fase 2: Análisis preliminar
        a_vals, e_vals = zip(*valid_results)
        a_array = np.array(a_vals)
        e_array = np.array(e_vals)

        # Estimación inicial con polinomio cuadrático
        try:
            coeffs = np.polyfit(a_array, e_array, 2)
            a_opt_initial = -coeffs[1] / (2 * coeffs[0])
            e_min_initial = np.polyval(coeffs, a_opt_initial)

            print(".6f"
        except:
            # Fallback: mínimo de puntos disponibles
            min_idx = np.argmin(e_array)
            a_opt_initial = a_array[min_idx]
            e_min_initial = e_array[min_idx]
            print(".6f"
        # Fase 3: Refinamiento local
        print("🔧 Fase 3: Refinamiento local")
        refinement_range = 0.05  # ±0.05 Å
        refinement_points = np.linspace(
            max(a_range[0], a_opt_initial - refinement_range),
            min(a_range[1], a_opt_initial + refinement_range),
            5
        )

        refinement_results = []
        for a in refinement_points:
            if a in a_array:  # Ya calculado
                existing_e = e_array[a_array == a][0]
                refinement_results.append((a, existing_e))
            else:
                try:
                    energy = await energy_calculator(a)
                    refinement_results.append((a, energy))
                    print(".4f"                except Exception as e:
                    print(".4f"                    refinement_results.append((a, np.nan))

        # Combinar todos los resultados
        all_results = valid_results + [(a, e) for a, e in refinement_results
                                      if np.isfinite(e) and a not in a_array]

        if len(all_results) < 3:
            return {'error': 'Insuficientes puntos para ajuste final'}

        # Fase 4: Ajuste final
        print("📊 Fase 4: Análisis final y ajuste")
        final_a, final_e = zip(*all_results)
        final_a_array = np.array(final_a)
        final_e_array = np.array(final_e)

        # Ajuste cuadrático final
        try:
            coeffs, cov = np.polyfit(final_a_array, final_e_array, 2, cov=True)
            A, B, C = coeffs
            a_opt = -B / (2 * A)
            e_min = np.polyval(coeffs, a_opt)

            # Calcular incertidumbre
            var_a = cov[1, 1] / (2 * A)**2 + (B**2 / (2 * A**4)) * cov[0, 0]
            sigma_a = np.sqrt(var_a) if var_a > 0 else np.nan

            # Calidad del ajuste
            e_fit = np.polyval(coeffs, final_a_array)
            ss_res = np.sum((final_e_array - e_fit)**2)
            ss_tot = np.sum((final_e_array - np.mean(final_e_array))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            computation_time = time.perf_counter() - start_time

            result = {
                'a_opt': float(a_opt),
                'e_min': float(e_min),
                'sigma_a': float(sigma_a) if np.isfinite(sigma_a) else None,
                'r2': float(r2),
                'fit_coefficients': coeffs.tolist(),
                'all_points': all_results,
                'n_total_points': len(all_results),
                'computation_time': computation_time,
                'success': True
            }

            print("✅ Optimización completada:"            print(".6f"            print(".6f"            print(".4f"            print(".1f"
            return result

        except Exception as e:
            # Fallback: mínimo observado
            min_idx = np.argmin(final_e_array)
            a_opt = final_a_array[min_idx]
            e_min = final_e_array[min_idx]

            return {
                'a_opt': float(a_opt),
                'e_min': float(e_min),
                'sigma_a': None,
                'r2': 0.0,
                'fit_coefficients': None,
                'all_points': all_results,
                'n_total_points': len(all_results),
                'computation_time': time.perf_counter() - start_time,
                'success': True,
                'warning': f'Ajuste falló: {e}, usando mínimo observado'
            }


class AdaptiveConvergenceController:
    """Controlador adaptativo de convergencia."""

    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self.convergence_history = []

    def should_stop_early(self, current_points: List[ConvergencePoint],
                         threshold: float) -> Tuple[bool, str]:
        """Determina si se debe detener temprano basado en convergencia."""

        if len(current_points) < 3:
            return False, "Insuficientes puntos"

        # Ordenar por parámetro
        sorted_points = sorted(current_points, key=lambda p: p.parameter)
        energies = [p.energy for p in sorted_points]

        # Verificar convergencia en últimas N puntos
        n_check = min(3, len(energies))
        recent_energies = energies[-n_check:]

        if all(np.isfinite(e) for e in recent_energies):
            energy_range = max(recent_energies) - min(recent_energies)
            if energy_range <= threshold:
                return True, f"Convergencia alcanzada (ΔE={energy_range:.2e} < {threshold:.2e})"

        # Verificar si el cambio es muy pequeño
        if len(energies) >= 2:
            last_change = abs(energies[-1] - energies[-2])
            if last_change < threshold * 0.1:
                return True, f"Cambio mínimo detectado (ΔE={last_change:.2e})"

        return False, "Continuar"

    def adapt_threshold(self, current_points: List[ConvergencePoint]) -> float:
        """Adapta threshold basado en historia de convergencia."""

        if not self.convergence_history:
            return self.config.conv_tol

        # Lógica simple: reducir threshold si converge rápido
        recent_thresholds = self.convergence_history[-3:]
        avg_threshold = np.mean(recent_thresholds)

        # Si los últimos cálculos convergieron rápido, ser más estricto
        if len(current_points) >= 3:
            energies = [p.energy for p in current_points[-3:]]
            if all(np.isfinite(e) for e in energies):
                energy_std = np.std(energies)
                if energy_std < avg_threshold:
                    return max(avg_threshold * 0.5, 1e-6)  # Más estricto

        return avg_threshold

    def update_history(self, threshold_used: float, converged: bool):
        """Actualiza historia de convergencia."""
        self.convergence_history.append(threshold_used)
        # Mantener solo últimos 10
        if len(self.convergence_history) > 10:
            self.convergence_history = self.convergence_history[-10:]


# Funciones de utilidad
def quadratic_fit(a_vals: np.ndarray, e_vals: np.ndarray) -> Dict[str, Any]:
    """Ajuste cuadrático para optimización de lattice."""
    mask = np.isfinite(a_vals) & np.isfinite(e_vals)
    x = a_vals[mask]
    y = e_vals[mask]

    if len(x) < 3:
        return {'error': 'Insuficientes puntos para ajuste cuadrático'}

    coeffs, cov = np.polyfit(x, y, 2, cov=True)
    A, B, C = coeffs
    a_opt = -B / (2 * A)
    e_min = np.polyval(coeffs, a_opt)

    # Calidad del ajuste
    yfit = np.polyval(coeffs, x)
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Incertidumbre
    dA = B / (2 * A**2)
    dB = -1.0 / (2 * A)
    varA, varB, covAB = cov[0,0], cov[1,1], cov[0,1]
    var_aopt = dA**2 * varA + dB**2 * varB + 2*dA*dB*covAB
    sigma_aopt = np.sqrt(abs(var_aopt)) if var_aopt >= 0 else np.nan

    return {
        'A': A, 'B': B, 'C': C,
        'a_opt': a_opt,
        'e_min': e_min,
        'r2': r2,
        'sigma_aopt': sigma_aopt,
        'cov': cov.tolist()
    }


def find_convergence_point(points: List[ConvergencePoint],
                          threshold: float,
                          parameter_name: str = "parameter") -> Optional[float]:
    """Encuentra punto de convergencia en lista de puntos."""

    if len(points) < 2:
        return None

    sorted_points = sorted(points, key=lambda p: p.parameter)
    energies = np.array([p.energy for p in sorted_points])

    # Encontrar energía mínima
    emin = np.min(energies)
    emin_idx = np.argmin(energies)

    # Buscar primer punto que esté dentro del threshold
    for i, point in enumerate(sorted_points):
        if abs(point.energy - emin) <= threshold:
            # Verificar que puntos subsiguientes también converjan
            subsequent = energies[i:]
            if len(subsequent) >= 2 and np.max(subsequent) - np.min(subsequent) <= threshold:
                return point.parameter

    return None