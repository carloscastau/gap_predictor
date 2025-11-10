# src/workflow/stages/bands_stage.py
"""Stage de cálculo de estructura de bandas."""

from typing import Dict, Any, List
import time

from .base import PipelineStage, StageResult
from ...core.calculator import DFTCalculator, CellParameters


class BandStructureStage(PipelineStage):
    """Stage de cálculo de estructura de bandas."""

    def __init__(self, config):
        super().__init__(config, "bands")

    def get_dependencies(self) -> List[str]:
        return ["cutoff", "kmesh", "lattice"]  # Depende de todos los parámetros óptimos

    def validate_inputs(self, previous_results: Dict[str, StageResult]) -> bool:
        required_stages = self.get_dependencies()
        for stage in required_stages:
            if stage not in previous_results or not previous_results[stage].success:
                return False
        return True

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        """Ejecuta cálculo de estructura de bandas."""
        start_time = time.perf_counter()

        try:
            self.logger.info("Starting band structure calculation")

            # Obtener parámetros óptimos de stages anteriores
            cutoff_result = previous_results["cutoff"]
            kmesh_result = previous_results["kmesh"]
            lattice_result = previous_results["lattice"]

            optimal_cutoff = cutoff_result.data.get("optimal_value", self.config.cutoff_list[-1])
            optimal_kmesh = kmesh_result.data.get("optimal_kmesh", self.config.kmesh_list[-1])
            optimal_lattice = lattice_result.data.get("a_opt", self.config.lattice_constant)

            self.logger.info(f"Using optimal parameters: cutoff={optimal_cutoff} Ry, "
                           f"k-mesh={optimal_kmesh}, lattice={optimal_lattice:.4f} Å")

            # Crear calculadora y parámetros de celda
            calculator = DFTCalculator(self.config)
            cell_params = CellParameters(
                lattice_constant=optimal_lattice,
                x_ga=self.config.x_ga,
                cutoff=optimal_cutoff,
                kmesh=optimal_kmesh,
                basis=self.config.basis_set,
                pseudo=self.config.pseudopotential,
                xc=self.config.xc_functional,
                sigma_ha=self.config.sigma_ha,
                conv_tol=1e-8
            )

            # Calcular estructura de bandas
            try:
                # Crear estructura pymatgen para path de alta simetría
                from pymatgen.core import Structure, Lattice

                # Celda convencional FCC para GaAs
                lattice = Lattice.cubic(optimal_lattice)
                structure = Structure(lattice, ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]])

                # Calcular bandas
                bands_result = await calculator.calculate_band_structure(cell_params, structure)

                if 'error' in bands_result:
                    self.logger.error(f"Band structure calculation failed: {bands_result['error']}")
                    return self._create_result(False, bands_result, time.perf_counter() - start_time,
                                              bands_result['error'])

                # Calcular DOS si está habilitado
                dos_result = None
                if hasattr(self.config, 'calculate_dos') and self.config.calculate_dos:
                    try:
                        # Obtener kmf del cálculo de bandas (esto requeriría modificación del calculator)
                        dos_result = {'message': 'DOS calculation not implemented in this version'}
                    except Exception as e:
                        self.logger.warning(f"DOS calculation failed: {e}")

                # Combinar resultados
                result_data = {
                    'bands_data': bands_result,
                    'dos_data': dos_result,
                    'calculation_parameters': {
                        'lattice_constant': optimal_lattice,
                        'cutoff': optimal_cutoff,
                        'kmesh': optimal_kmesh,
                        'basis': self.config.basis_set,
                        'xc': self.config.xc_functional
                    }
                }

                # Extraer gap si está disponible
                if 'converged' in bands_result and bands_result['converged']:
                    # Aquí se extraería información del gap de los resultados
                    result_data['gap_info'] = {
                        'message': 'Gap extraction requires kmf object from SCF calculation'
                    }

                duration = time.perf_counter() - start_time
                self.logger.info(f"Band structure calculation completed in {duration:.2f}s")

                return self._create_result(True, result_data, duration)

            except ImportError as e:
                error_msg = f"Required libraries not available: {e}"
                self.logger.error(error_msg)
                return self._create_result(False, {'error': error_msg}, time.perf_counter() - start_time, error_msg)

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(f"Band structure stage failed: {e}")
            return self._create_result(False, {}, duration, str(e))