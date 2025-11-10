# src/workflow/stages/lattice_stage.py
"""Stage de optimización de parámetro de red."""

from typing import Dict, Any, List
import time

from .base import OptimizationStage, StageResult
from ...core.calculator import DFTCalculator, CellParameters


class LatticeOptimizationStage(OptimizationStage):
    """Stage de optimización de parámetro de red."""

    def __init__(self, config):
        super().__init__(config, "lattice")

    def get_dependencies(self) -> List[str]:
        return ["cutoff", "kmesh"]  # Depende de cutoff y kmesh óptimos

    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        required_stages = self.get_dependencies()
        for stage in required_stages:
            if stage not in previous_results or not previous_results[stage].success:
                return False
        return True

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta optimización de lattice."""
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting lattice optimization")

            # Obtener parámetros óptimos de stages anteriores
            cutoff_result = previous_results["cutoff"]
            kmesh_result = previous_results["kmesh"]

            optimal_cutoff = cutoff_result.data.get("optimal_value", self.config.cutoff_list[-1])
            optimal_kmesh = kmesh_result.data.get("optimal_kmesh", self.config.kmesh_list[-1])

            self.logger.info(f"Using optimal cutoff: {optimal_cutoff} Ry")
            self.logger.info(f"Using optimal k-mesh: {optimal_kmesh}")

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
            bounds = (5.4, 5.8)  # Rango físico para GaAs
            result_data = await self.execute_optimization(
                optimizer_func=energy_func,
                bounds=bounds,
                initial_guess=self.config.lattice_constant
            )

            duration = time.perf_counter() - start_time

            if 'error' in result_data:
                self.logger.error(f"Lattice optimization failed: {result_data['error']}")
                return self._create_result(False, result_data, duration, result_data['error'])

            # Loggear resultados
            a_opt = result_data.get('a_opt')
            e_min = result_data.get('e_min')
            r2 = result_data.get('r2', 0)

            if a_opt and e_min:
                self.logger.info(f"Lattice optimization completed. a_opt: {a_opt:.6f} Å, "
                               f"E_min: {e_min:.8f} Ha, R²: {r2:.4f}")
            else:
                self.logger.warning("Lattice optimization completed but optimal values not found")

            return self._create_result(True, result_data, duration)

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(f"Lattice optimization stage failed: {e}")
            return self._create_result(False, {}, duration, str(e))