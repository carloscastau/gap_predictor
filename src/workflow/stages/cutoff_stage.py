# src/workflow/stages/cutoff_stage.py
"""Stage de convergencia de cutoff."""

from typing import Dict, Any, List
import time

from .base import ConvergenceStage, StageResult
from ...core.calculator import DFTCalculator, CellParameters


class CutoffConvergenceStage(ConvergenceStage):
    """Stage de convergencia de cutoff."""

    def __init__(self, config):
        super().__init__(config, "cutoff", "cutoff")

    def get_dependencies(self) -> List[str]:
        return []  # No dependencies

    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        return True  # No dependencies

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta convergencia de cutoff."""
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting cutoff convergence analysis")

            # Función de energía para diferentes cutoffs
            async def energy_func(cutoff: float) -> float:
                calculator = DFTCalculator(self.config)
                cell_params = CellParameters(
                    lattice_constant=self.config.lattice_constant,
                    x_ga=self.config.x_ga,
                    cutoff=cutoff,
                    kmesh=self.config.kmesh_list[0],  # Usar primera malla k
                    basis=self.config.basis_set,
                    pseudo=self.config.pseudopotential,
                    xc=self.config.xc_functional,
                    sigma_ha=self.config.sigma_ha,
                    conv_tol=1e-8
                )

                result = await calculator.calculate_energy(cell_params)
                return result.energy if result.converged else float('nan')

            # Ejecutar análisis de convergencia
            result_data = await self.execute_convergence_analysis(
                parameter_values=self.config.cutoff_list,
                energy_calculator=energy_func,
                threshold=1e-4  # Ha
            )

            duration = time.perf_counter() - start_time

            if 'error' in result_data:
                self.logger.error(f"Cutoff convergence failed: {result_data['error']}")
                return self._create_result(False, result_data, duration, result_data['error'])

            # Loggear resultados
            optimal_cutoff = result_data.get('optimal_value')
            if optimal_cutoff:
                self.logger.info(f"Cutoff convergence completed. Optimal: {optimal_cutoff} Ry")
            else:
                self.logger.warning("Cutoff convergence did not find optimal value")

            return self._create_result(True, result_data, duration)

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(f"Cutoff stage failed: {e}")
            return self._create_result(False, {}, duration, str(e))