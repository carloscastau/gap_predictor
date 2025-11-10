# src/workflow/stages/kmesh_stage.py
"""Stage de convergencia de k-mesh."""

from typing import Dict, Any, List
import time

from .base import ConvergenceStage, StageResult
from ...core.calculator import DFTCalculator, CellParameters


class KMeshConvergenceStage(ConvergenceStage):
    """Stage de convergencia de k-mesh."""

    def __init__(self, config):
        super().__init__(config, "kmesh", "kmesh")

    def get_dependencies(self) -> List[str]:
        return ["cutoff"]  # Depende del cutoff óptimo

    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        return "cutoff" in previous_results and previous_results["cutoff"].success

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta convergencia de k-mesh."""
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting k-mesh convergence analysis")

            # Obtener cutoff óptimo del stage anterior
            cutoff_result = previous_results["cutoff"]
            optimal_cutoff = cutoff_result.data.get("optimal_value")

            if optimal_cutoff is None:
                # Fallback al último valor de la lista
                optimal_cutoff = self.config.cutoff_list[-1]
                self.logger.warning(f"Using fallback cutoff: {optimal_cutoff} Ry")

            # Función de energía para diferentes k-mesh
            async def energy_func(kmesh_tuple) -> float:
                calculator = DFTCalculator(self.config)
                cell_params = CellParameters(
                    lattice_constant=self.config.lattice_constant,
                    x_ga=self.config.x_ga,
                    cutoff=optimal_cutoff,
                    kmesh=kmesh_tuple,
                    basis=self.config.basis_set,
                    pseudo=self.config.pseudopotential,
                    xc=self.config.xc_functional,
                    sigma_ha=self.config.sigma_ha,
                    conv_tol=1e-8
                )

                result = await calculator.calculate_energy(cell_params)
                return result.energy if result.converged else float('nan')

            # Convertir kmesh_list a valores para análisis
            # Usar número total de k-points como parámetro
            kmesh_values = [k[0] * k[1] * k[2] for k in self.config.kmesh_list]

            # Ejecutar análisis de convergencia
            result_data = await self.execute_convergence_analysis(
                parameter_values=kmesh_values,
                energy_calculator=energy_func,
                threshold=1e-5  # Ha, más estricto que cutoff
            )

            duration = time.perf_counter() - start_time

            if 'error' in result_data:
                self.logger.error(f"k-mesh convergence failed: {result_data['error']}")
                return self._create_result(False, result_data, duration, result_data['error'])

            # Convertir de vuelta a tupla de k-mesh
            optimal_nkpts = result_data.get('optimal_value')
            optimal_kmesh = None

            if optimal_nkpts:
                # Encontrar la tupla de k-mesh correspondiente
                for kmesh in self.config.kmesh_list:
                    if kmesh[0] * kmesh[1] * kmesh[2] == optimal_nkpts:
                        optimal_kmesh = kmesh
                        break

            if optimal_kmesh:
                result_data['optimal_kmesh'] = optimal_kmesh
                self.logger.info(f"k-mesh convergence completed. Optimal: {optimal_kmesh} ({optimal_nkpts} k-points)")
            else:
                # Fallback
                optimal_kmesh = self.config.kmesh_list[-1]
                result_data['optimal_kmesh'] = optimal_kmesh
                self.logger.warning(f"Using fallback k-mesh: {optimal_kmesh}")

            return self._create_result(True, result_data, duration)

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(f"k-mesh stage failed: {e}")
            return self._create_result(False, {}, duration, str(e))