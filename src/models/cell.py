# src/models/cell.py
"""Modelos de datos para celdas unitarias."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class CellParameters:
    """Parámetros de celda unitaria."""
    lattice_constant: float
    x_ga: float = 0.25
    cutoff: float = 100.0
    kmesh: Tuple[int, int, int] = (4, 4, 4)
    basis: str = "gth-dzvp"
    pseudo: str = "gth-pbe"
    xc: str = "PBE"
    sigma_ha: float = 0.01
    conv_tol: float = 1e-8

    @property
    def estimated_memory(self) -> float:
        """Estima uso de memoria en MB."""
        nkpts = self.kmesh[0] * self.kmesh[1] * self.kmesh[2]
        return 200 + nkpts * 50  # MB

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Retorna vectores de red."""
        a = self.lattice_constant
        return np.array([
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a]
        ])

    @property
    def atomic_positions(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Retorna posiciones atómicas para GaAs."""
        # Estructura zincblende
        r_ga_cart = self.x_ga * np.array([self.lattice_constant] * 3)
        return [
            ("As", (0.0, 0.0, 0.0)),
            ("Ga", tuple(r_ga_cart))
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'lattice_constant': self.lattice_constant,
            'x_ga': self.x_ga,
            'cutoff': self.cutoff,
            'kmesh': self.kmesh,
            'basis': self.basis,
            'pseudo': self.pseudo,
            'xc': self.xc,
            'sigma_ha': self.sigma_ha,
            'conv_tol': self.conv_tol
        }


@dataclass
class CellModel:
    """Modelo completo de celda unitaria."""

    parameters: CellParameters
    lattice_vectors: np.ndarray = field(init=False)
    reciprocal_vectors: np.ndarray = field(init=False)
    volume: float = field(init=False)
    atomic_positions: List[Tuple[str, Tuple[float, float, float]]] = field(init=False)

    def __post_init__(self):
        """Inicializa propiedades derivadas."""
        self.lattice_vectors = self.parameters.lattice_vectors
        self.reciprocal_vectors = self._calculate_reciprocal_vectors()
        self.volume = self._calculate_volume()
        self.atomic_positions = self.parameters.atomic_positions

    def _calculate_reciprocal_vectors(self) -> np.ndarray:
        """Calcula vectores recíprocos."""
        return 2 * np.pi * np.linalg.inv(self.lattice_vectors.T)

    def _calculate_volume(self) -> float:
        """Calcula volumen de la celda."""
        return abs(np.linalg.det(self.lattice_vectors))

    @property
    def n_atoms(self) -> int:
        """Número de átomos en la celda."""
        return len(self.atomic_positions)

    @property
    def chemical_formula(self) -> str:
        """Fórmula química."""
        atoms = {}
        for symbol, _ in self.atomic_positions:
            atoms[symbol] = atoms.get(symbol, 0) + 1

        return "".join(f"{symbol}{count}" for symbol, count in atoms.items())

    def get_kpoints(self, kmesh: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Genera puntos k para una malla dada."""
        if kmesh is None:
            kmesh = self.parameters.kmesh

        # Generar malla uniforme
        kx = np.linspace(0, 1, kmesh[0], endpoint=False)
        ky = np.linspace(0, 1, kmesh[1], endpoint=False)
        kz = np.linspace(0, 1, kmesh[2], endpoint=False)

        kpoints = []
        for i in kx:
            for j in ky:
                for k in kz:
                    kpoints.append([i, j, k])

        return np.array(kpoints)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable."""
        return {
            'parameters': self.parameters.to_dict(),
            'lattice_vectors': self.lattice_vectors.tolist(),
            'reciprocal_vectors': self.reciprocal_vectors.tolist(),
            'volume': self.volume,
            'atomic_positions': self.atomic_positions,
            'n_atoms': self.n_atoms,
            'chemical_formula': self.chemical_formula
        }