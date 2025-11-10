# tests/unit/test_models.py
"""Tests para modelos de datos."""

import pytest
import numpy as np

from src.models.cell import CellParameters, CellModel
from src.models.kpoints import KMesh, create_kmesh_from_tuple
from src.models.results import CalculationResult, ConvergenceResult


class TestCellParameters:
    """Tests para CellParameters."""

    def test_cell_parameters_creation(self):
        """Test creación de parámetros de celda."""
        params = CellParameters(
            lattice_constant=5.653,
            x_ga=0.25,
            cutoff=100.0,
            kmesh=(4, 4, 4)
        )

        assert params.lattice_constant == 5.653
        assert params.x_ga == 0.25
        assert params.cutoff == 100.0
        assert params.kmesh == (4, 4, 4)

    def test_lattice_vectors(self):
        """Test cálculo de vectores de red."""
        params = CellParameters(lattice_constant=5.0)

        vectors = params.lattice_vectors
        expected = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]
        ])

        np.testing.assert_array_equal(vectors, expected)

    def test_atomic_positions(self):
        """Test posiciones atómicas."""
        params = CellParameters(lattice_constant=5.653, x_ga=0.25)

        positions = params.atomic_positions

        assert len(positions) == 2
        assert positions[0] == ("As", (0.0, 0.0, 0.0))

        # Verificar posición de Ga
        ga_pos = positions[1][1]
        expected_ga = 0.25 * 5.653
        assert abs(ga_pos[0] - expected_ga) < 1e-10
        assert abs(ga_pos[1] - expected_ga) < 1e-10
        assert abs(ga_pos[2] - expected_ga) < 1e-10

    def test_memory_estimation(self):
        """Test estimación de memoria."""
        params = CellParameters(kmesh=(4, 4, 4))

        memory_mb = params.estimated_memory
        assert memory_mb > 0
        assert isinstance(memory_mb, float)


class TestCellModel:
    """Tests para CellModel."""

    def test_cell_model_creation(self):
        """Test creación de modelo de celda."""
        params = CellParameters(lattice_constant=5.0)
        model = CellModel(params)

        assert model.parameters == params
        assert model.n_atoms == 2
        assert model.chemical_formula == "AsGa"

    def test_reciprocal_vectors(self):
        """Test cálculo de vectores recíprocos."""
        params = CellParameters(lattice_constant=5.0)
        model = CellModel(params)

        # Para celda cúbica, b = 2π/a * I
        expected_b = 2 * np.pi / 5.0
        expected = np.eye(3) * expected_b

        np.testing.assert_array_almost_equal(model.reciprocal_vectors, expected)

    def test_volume_calculation(self):
        """Test cálculo de volumen."""
        params = CellParameters(lattice_constant=5.0)
        model = CellModel(params)

        expected_volume = 5.0 ** 3
        assert abs(model.volume - expected_volume) < 1e-10


class TestKMesh:
    """Tests para KMesh."""

    def test_kmesh_creation(self):
        """Test creación de malla k."""
        kmesh = KMesh(4, 4, 4)

        assert kmesh.nx == 4
        assert kmesh.ny == 4
        assert kmesh.nz == 4
        assert kmesh.total_points == 64
        assert kmesh.as_tuple() == (4, 4, 4)

    def test_kmesh_from_tuple(self):
        """Test creación desde tupla."""
        kmesh = create_kmesh_from_tuple((3, 3, 3))

        assert kmesh.nx == 3
        assert kmesh.ny == 3
        assert kmesh.nz == 3

    def test_kmesh_from_string(self):
        """Test creación desde string."""
        from src.models.kpoints import create_kmesh_from_string

        kmesh = create_kmesh_from_string("4x4x4")

        assert kmesh.nx == 4
        assert kmesh.ny == 4
        assert kmesh.nz == 4


class TestCalculationResult:
    """Tests para CalculationResult."""

    def test_calculation_result_creation(self):
        """Test creación de resultado de cálculo."""
        result = CalculationResult(
            task_id="test_task",
            success=True,
            energy=-10.5,
            converged=True,
            n_iterations=50,
            computation_time=2.5
        )

        assert result.task_id == "test_task"
        assert result.success is True
        assert result.energy == -10.5
        assert result.converged is True
        assert result.is_valid is True

    def test_energy_conversion(self):
        """Test conversión de energía a eV."""
        result = CalculationResult(
            task_id="test",
            success=True,
            energy=-0.5,  # Ha
            converged=True
        )

        expected_ev = -0.5 * 27.211386245988
        assert abs(result.energy_ev - expected_ev) < 1e-10

    def test_invalid_result(self):
        """Test resultado inválido."""
        result = CalculationResult(
            task_id="test",
            success=False,
            energy=np.nan,
            converged=False
        )

        assert result.is_valid is False


class TestConvergenceResult:
    """Tests para ConvergenceResult."""

    def test_convergence_result_creation(self):
        """Test creación de resultado de convergencia."""
        result = ConvergenceResult(
            parameter_name="cutoff",
            converged=True,
            optimal_value=120.0,
            points_analyzed=5,
            fit_quality=0.95
        )

        assert result.parameter_name == "cutoff"
        assert result.converged is True
        assert result.optimal_value == 120.0
        assert result.has_sufficient_points is True

    def test_convergence_ratio(self):
        """Test cálculo de ratio de convergencia."""
        result = ConvergenceResult(
            parameter_name="cutoff",
            converged=True,
            parameter_values=[80, 100, 120],
            energies=[-10.0, -10.2, -10.25]
        )

        # Debería calcular ratio basado en reducción de error
        assert isinstance(result.convergence_ratio, float)