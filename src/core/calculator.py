# src/core/calculator.py
"""Calculadora DFT optimizada con gestión de memoria."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from ..config.settings import PreconvergenceConfig
from .parallel import MemoryMonitor


@dataclass
class CellParameters:
    """Parámetros de celda unitaria."""
    lattice_constant: float
    x_ga: float
    cutoff: float
    kmesh: Tuple[int, int, int]
    basis: str
    pseudo: str
    xc: str
    sigma_ha: float
    conv_tol: float

    @property
    def estimated_memory(self) -> float:
        """Estima uso de memoria en MB."""
        nkpts = self.kmesh[0] * self.kmesh[1] * self.kmesh[2]
        # Estimación simplificada
        return 200 + nkpts * 50  # MB


@dataclass
class EnergyResult:
    """Resultado de cálculo de energía."""
    energy: float
    converged: bool
    n_iterations: Optional[int]
    memory_peak: float
    computation_time: float


class DFTCalculator:
    """Calculadora DFT optimizada con gestión de memoria."""

    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self._setup_pyscf()

    def _setup_pyscf(self):
        """Configuración optimizada de PySCF."""
        try:
            from pyscf import lib as pyscf_lib

            # Configurar límites de memoria
            pyscf_lib.param.MAX_MEMORY = self.config.memory_limit_gb * 1024**3

            # Configurar paralelismo interno
            pyscf_lib.num_threads(1)  # Dejar paralelismo al nivel superior

            # Configurar tolerancias
            pyscf_lib.param.TOLERANCE = 1e-10

        except ImportError:
            # PySCF no disponible, usar simulación
            pass

    @MemoryMonitor.track_usage
    async def calculate_energy(self, cell_params: CellParameters) -> EnergyResult:
        """Calcula energía con monitoreo de memoria."""
        start_time = time.perf_counter()

        # Verificar límites de memoria antes de calcular
        if not self.memory_monitor.check_available_memory(cell_params.estimated_memory):
            raise MemoryError("Memoria insuficiente para el cálculo")

        try:
            # Construir celda
            cell = await self._build_cell_async(cell_params)

            # Ejecutar cálculo DFT
            with self.memory_monitor.memory_context("SCF"):
                kmf, converged = await self._run_scf_async(cell, cell_params.kmesh)

            # Extraer resultados
            energy = self._extract_energy(kmf) if converged else np.nan

            computation_time = time.perf_counter() - start_time

            return EnergyResult(
                energy=energy,
                converged=converged,
                n_iterations=getattr(kmf, 'niter', None) if hasattr(kmf, 'niter') else None,
                memory_peak=self.memory_monitor.get_peak_usage(),
                computation_time=computation_time
            )

        except Exception as e:
            computation_time = time.perf_counter() - start_time
            return EnergyResult(
                energy=np.nan,
                converged=False,
                n_iterations=None,
                memory_peak=self.memory_monitor.get_peak_usage(),
                computation_time=computation_time
            )

    async def _build_cell_async(self, cell_params: CellParameters):
        """Construye celda de forma asíncrona."""
        try:
            from pyscf.pbc import gto
            from pyscf import lib as pyscf_lib

            # Definición de celda GaAs
            a_A = cell_params.lattice_constant
            x_ga = cell_params.x_ga

            a1 = np.array([a_A, 0.0, 0.0])
            a2 = np.array([0.0, a_A, 0.0])
            a3 = np.array([0.0, 0.0, a_A])

            r_ga_frac = np.array([x_ga, x_ga, x_ga])
            r_ga_cart = r_ga_frac @ np.vstack([a1, a2, a3])

            cell = gto.Cell()
            cell.unit = "A"
            cell.a = np.vstack([a1, a2, a3])
            cell.atom = [("As", (0.0, 0.0, 0.0)), ("Ga", tuple(r_ga_cart))]
            cell.basis = {"Ga": cell_params.basis, "As": cell_params.basis}

            # Pseudopotenciales
            try:
                cell.pseudo = {"Ga": cell_params.pseudo, "As": cell_params.pseudo}
            except:
                cell.pseudo = None

            cell.ke_cutoff = pyscf_lib.param.RY_TO_HARTREE * cell_params.cutoff
            cell.precision = cell_params.conv_tol
            cell.verbose = 0
            cell.exp_to_discard = 0.1

            cell.build()
            return cell

        except ImportError:
            # Simular construcción de celda
            return self._simulate_cell_build(cell_params)

    def _simulate_cell_build(self, cell_params: CellParameters):
        """Simula construcción de celda para desarrollo."""
        class MockCell:
            def __init__(self, params):
                self.params = params
                self.a = np.eye(3) * params.lattice_constant
                self.atom = [("As", (0,0,0)), ("Ga", (params.x_ga, params.x_ga, params.x_ga))]

        return MockCell(cell_params)

    async def _run_scf_async(self, cell, kmesh: Tuple[int, int, int]):
        """Ejecuta SCF de forma asíncrona."""
        try:
            from pyscf.pbc import dft
            from pyscf.scf import addons as scf_addons

            # Crear k-points
            kpts = cell.make_kpts(kmesh)

            # Configurar cálculo DFT
            kmf = dft.KRKS(cell, kpts=kpts)
            kmf.xc = self.config.xc_functional
            kmf.conv_tol = self.config.conv_tol if hasattr(self.config, 'conv_tol') else 1e-8
            kmf.max_cycle = 80

            # Smearing
            kmf = scf_addons.smearing_(kmf, sigma=self.config.sigma_ha, method="fermi")
            kmf = scf_addons.remove_linear_dep_(kmf)

            # Ejecutar
            energy = kmf.kernel()

            return kmf, kmf.converged

        except ImportError:
            # Simular cálculo SCF
            return self._simulate_scf(cell, kmesh)

    def _simulate_scf(self, cell, kmesh):
        """Simula cálculo SCF para desarrollo."""
        import random

        class MockKMF:
            def __init__(self):
                self.converged = random.random() > 0.1  # 90% convergencia
                self.niter = random.randint(20, 80) if self.converged else 80
                self.e_tot = -10.5 + random.gauss(0, 0.01) if self.converged else np.nan

        return MockKMF(), MockKMF().converged

    def _extract_energy(self, kmf) -> float:
        """Extrae energía del objeto kmf."""
        # Intentar diferentes formas de obtener energía
        if hasattr(kmf, 'e_tot'):
            return float(kmf.e_tot)
        elif hasattr(kmf, 'energy_tot'):
            return float(kmf.energy_tot())
        else:
            return np.nan

    async def calculate_band_structure(self, cell_params: CellParameters,
                                     structure) -> Dict[str, Any]:
        """Calcula estructura de bandas."""
        try:
            from pymatgen.core.surface import HighSymmKpath
            from pyscf.pbc import dft

            # Construir celda
            cell = await self._build_cell_async(cell_params)

            # Ejecutar SCF ground state
            kmf, converged = await self._run_scf_async(cell, cell_params.kmesh)

            if not converged:
                return {'error': 'SCF no convergió'}

            # Calcular bandas
            kpath = HighSymmKpath(structure)
            kpts_labels = kpath.kpath["kpoints"]
            path = kpath.kpath["path"]

            # Generar k-points para path
            kpts_frac = []
            labels = []
            for seg in path:
                start, end = seg
                k0 = np.array(kpts_labels[start])
                k1 = np.array(kpts_labels[end])
                for i in range(50):  # 50 puntos por segmento
                    t = i / 49
                    kpts_frac.append((1-t)*k0 + t*k1)
                    labels.append("")

            kpts_cart = (np.array(kpts_frac) @ cell.reciprocal_vectors())
            bands = kmf.get_bands(kpts_cart)[0]

            return {
                'bands': bands,
                'kpts_frac': kpts_frac,
                'labels': labels,
                'converged': converged
            }

        except ImportError:
            return {'error': 'PySCF/pymatgen no disponibles'}

    async def calculate_dos(self, kmf, sigma_ev: float = 0.1) -> Dict[str, Any]:
        """Calcula densidad de estados."""
        try:
            # Extraer eigenvalores
            mo_e = getattr(kmf, "mo_energy", [])
            if not mo_e:
                return {'error': 'No se pudieron obtener eigenvalores'}

            evals = np.hstack([np.array(e) for e in mo_e])
            evals_ev = evals * 27.211386245988  # Ha to eV

            # Rango de energía
            e_min, e_max = np.min(evals_ev) - 3.0, np.max(evals_ev) + 3.0
            E = np.linspace(e_min, e_max, 2048)
            dos = np.zeros_like(E)

            # Calcular DOS con Gaussiana
            w = sigma_ev
            for e in evals_ev:
                dos += np.exp(-0.5 * ((E - e) / w) ** 2) / (w * np.sqrt(2 * np.pi))

            return {
                'energy_ev': E,
                'dos': dos,
                'sigma_ev': sigma_ev
            }

        except Exception as e:
            return {'error': str(e)}


# Funciones de utilidad
def zeroT_energy_from_smearing(kmf) -> float:
    """Extrae energía a T=0 desde cálculo con smearing."""
    e_tot = getattr(kmf, "e_tot", np.nan)
    e_free = getattr(kmf, "e_free", None)
    return 0.5 * (e_tot + e_free) if e_free is not None else e_tot


def create_cell_parameters(lattice_constant: float, x_ga: float, cutoff: float,
                          kmesh: Tuple[int, int, int], basis: str, pseudo: str,
                          xc: str = "PBE", sigma_ha: float = 0.01,
                          conv_tol: float = 1e-8) -> CellParameters:
    """Crea parámetros de celda desde valores individuales."""
    return CellParameters(
        lattice_constant=lattice_constant,
        x_ga=x_ga,
        cutoff=cutoff,
        kmesh=kmesh,
        basis=basis,
        pseudo=pseudo,
        xc=xc,
        sigma_ha=sigma_ha,
        conv_tol=conv_tol
    )