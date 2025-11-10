# src/models/kpoints.py
"""Modelos de datos para puntos k."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class KMesh:
    """Malla de puntos k."""

    nx: int
    ny: int
    nz: int

    @property
    def total_points(self) -> int:
        """Número total de puntos k."""
        return self.nx * self.ny * self.nz

    @property
    def as_tuple(self) -> Tuple[int, int, int]:
        """Retorna como tupla."""
        return (self.nx, self.ny, self.nz)

    @property
    def as_list(self) -> List[int]:
        """Retorna como lista."""
        return [self.nx, self.ny, self.nz]

    @property
    def density(self) -> float:
        """Densidad de puntos k (puntos/Å⁻³)."""
        # Estimación simple
        return self.total_points ** (1/3) / 5.0  # Normalizado por parámetro de red típico

    def __str__(self) -> str:
        return f"{self.nx}x{self.ny}x{self.nz}"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
            'total_points': self.total_points,
            'density': self.density
        }


@dataclass
class KPointsModel:
    """Modelo completo de puntos k."""

    kmesh: KMesh
    kpoints: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    lattice_vectors: Optional[np.ndarray] = None

    def __post_init__(self):
        """Inicializa puntos k y pesos."""
        self.kpoints = self._generate_kpoints()
        self.weights = self._generate_weights()

    def _generate_kpoints(self) -> np.ndarray:
        """Genera puntos k para la malla."""
        kx = np.linspace(0, 1, self.kmesh.nx, endpoint=False)
        ky = np.linspace(0, 1, self.kmesh.ny, endpoint=False)
        kz = np.linspace(0, 1, self.kmesh.nz, endpoint=False)

        kpoints = []
        for i in kx:
            for j in ky:
                for k in kz:
                    kpoints.append([i, j, k])

        return np.array(kpoints)

    def _generate_weights(self) -> np.ndarray:
        """Genera pesos para los puntos k."""
        total_points = self.kmesh.total_points
        return np.full(total_points, 1.0 / total_points)

    def get_cartesian_kpoints(self) -> np.ndarray:
        """Convierte puntos k a coordenadas cartesianas."""
        if self.lattice_vectors is None:
            # Asumir celda cúbica unitaria
            b = 2 * np.pi * np.eye(3)
        else:
            b = 2 * np.pi * np.linalg.inv(self.lattice_vectors.T)

        return self.kpoints @ b

    def get_kpoint_distances(self) -> np.ndarray:
        """Calcula distancias entre puntos k consecutivos."""
        cart_kpoints = self.get_cartesian_kpoints()
        distances = []

        for i in range(1, len(cart_kpoints)):
            dist = np.linalg.norm(cart_kpoints[i] - cart_kpoints[i-1])
            distances.append(dist)

        return np.array(distances)

    @property
    def min_distance(self) -> float:
        """Distancia mínima entre puntos k."""
        distances = self.get_kpoint_distances()
        return np.min(distances) if len(distances) > 0 else 0.0

    @property
    def max_distance(self) -> float:
        """Distancia máxima entre puntos k."""
        distances = self.get_kpoint_distances()
        return np.max(distances) if len(distances) > 0 else 0.0

    @property
    def avg_distance(self) -> float:
        """Distancia promedio entre puntos k."""
        distances = self.get_kpoint_distances()
        return np.mean(distances) if len(distances) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable."""
        return {
            'kmesh': self.kmesh.to_dict(),
            'kpoints': self.kpoints.tolist(),
            'weights': self.weights.tolist(),
            'lattice_vectors': self.lattice_vectors.tolist() if self.lattice_vectors is not None else None,
            'cartesian_kpoints': self.get_cartesian_kpoints().tolist(),
            'min_distance': self.min_distance,
            'max_distance': self.max_distance,
            'avg_distance': self.avg_distance
        }


# Funciones de utilidad
def create_kmesh_from_tuple(kmesh_tuple: Tuple[int, int, int]) -> KMesh:
    """Crea KMesh desde tupla."""
    return KMesh(kmesh_tuple[0], kmesh_tuple[1], kmesh_tuple[2])


def create_kmesh_from_string(kmesh_str: str) -> KMesh:
    """Crea KMesh desde string (ej: '4x4x4')."""
    parts = kmesh_str.split('x')
    if len(parts) != 3:
        raise ValueError(f"Invalid kmesh string: {kmesh_str}")

    return KMesh(int(parts[0]), int(parts[1]), int(parts[2]))


def generate_gamma_centered_kmesh(n_kpts: int) -> KMesh:
    """Genera malla gamma-centrada con aproximadamente n_kpts puntos."""
    # Estimación simple: n_kpts ≈ n^3
    n = int(np.ceil(n_kpts ** (1/3)))
    return KMesh(n, n, n)


def get_optimal_kmesh_for_cutoff(cutoff: float, lattice_constant: float = 5.65) -> KMesh:
    """Estima malla k óptima para un cutoff dado."""
    # Regla empírica simple
    k_density = cutoff / (10 * lattice_constant)  # puntos/Å
    n = max(2, int(np.ceil(k_density * lattice_constant)))
    return KMesh(n, n, n)