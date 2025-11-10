# tests/integration/test_pipeline.py
"""Tests de integración para el pipeline completo."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from src.config.settings import get_fast_config
from src.workflow.pipeline import PreconvergencePipeline, PipelineResult


class TestPreconvergencePipeline:
    """Tests para el pipeline completo."""

    @pytest.fixture
    def fast_config(self):
        """Configuración rápida para tests."""
        return get_fast_config()

    @pytest.fixture
    def mock_calculator(self):
        """Mock de calculadora DFT."""
        with patch('src.core.calculator.DFTCalculator') as mock_calc:
            # Configurar mock para devolver resultados simulados
            mock_instance = Mock()
            mock_instance.calculate_energy.return_value = Mock(
                energy=-10.5,
                converged=True,
                n_iterations=25,
                memory_peak=512.0
            )
            mock_calc.return_value = mock_instance
            yield mock_calc

    @pytest.fixture
    def mock_optimizer(self):
        """Mock de optimizador."""
        with patch('src.core.optimizer.LatticeOptimizer') as mock_opt:
            mock_instance = Mock()
            mock_instance.optimize_lattice_constant.return_value = {
                'a_opt': 5.653,
                'e_min': -10.5,
                'success': True
            }
            mock_opt.return_value = mock_instance
            yield mock_opt

    @pytest.mark.asyncio
    async def test_pipeline_creation(self, fast_config):
        """Test creación del pipeline."""
        pipeline = PreconvergencePipeline(fast_config)

        assert pipeline.config == fast_config
        assert 'cutoff' in pipeline.stages
        assert 'kmesh' in pipeline.stages
        assert 'lattice' in pipeline.stages

    @pytest.mark.asyncio
    async def test_pipeline_execution_mock(self, fast_config, mock_calculator, mock_optimizer):
        """Test ejecución completa del pipeline con mocks."""
        pipeline = PreconvergencePipeline(fast_config)

        # Ejecutar pipeline
        result = await pipeline.execute()

        # Verificar resultado
        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert 'cutoff' in result.results
        assert 'kmesh' in result.results
        assert 'lattice' in result.results

    @pytest.mark.asyncio
    async def test_pipeline_with_resume(self, fast_config, tmp_path):
        """Test pipeline con reanudación desde checkpoint."""
        # Crear directorio de checkpoints simulado
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Crear checkpoint simulado
        import json
        checkpoint_data = {
            "stage": "cutoff",
            "timestamp": "2024-01-01T00:00:00",
            "status": "completed",
            "data": {"optimal_value": 120.0}
        }

        checkpoint_file = checkpoint_dir / "checkpoint_cutoff_completed.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        # Configurar config para usar directorio temporal
        fast_config.output_dir = tmp_path

        pipeline = PreconvergencePipeline(fast_config)

        # Verificar que el checkpoint se carga
        assert pipeline.checkpoint_manager is not None

    def test_stage_dependencies(self, fast_config):
        """Test dependencias entre stages."""
        pipeline = PreconvergencePipeline(fast_config)

        # Verificar dependencias
        cutoff_stage = pipeline.stages['cutoff']
        kmesh_stage = pipeline.stages['kmesh']
        lattice_stage = pipeline.stages['lattice']

        # Cutoff no tiene dependencias
        assert len(cutoff_stage.get_dependencies()) == 0

        # K-mesh depende de cutoff
        assert 'cutoff' in kmesh_stage.get_dependencies()

        # Lattice depende de cutoff y kmesh
        deps = lattice_stage.get_dependencies()
        assert 'cutoff' in deps
        assert 'kmesh' in deps

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, fast_config):
        """Test manejo de errores en pipeline."""
        pipeline = PreconvergencePipeline(fast_config)

        # Simular error en un stage
        with patch.object(pipeline.stages['cutoff'], 'execute') as mock_stage:
            mock_stage.side_effect = Exception("Test error")

            result = await pipeline.execute()

            # El pipeline debería manejar el error
            assert result.success is False

    def test_pipeline_result_structure(self):
        """Test estructura del resultado del pipeline."""
        results = {
            'cutoff': Mock(success=True, data={'optimal_value': 120.0}),
            'kmesh': Mock(success=True, data={'optimal_kmesh': (4, 4, 4)}),
            'lattice': Mock(success=True, data={'a_opt': 5.653})
        }

        result = PipelineResult(results=results, config=get_fast_config())

        assert result.success is True
        assert result.total_stages == 3
        assert result.completed_stages == 3
        assert result.failed_stages == 0

    def test_pipeline_partial_success(self):
        """Test resultado parcial del pipeline."""
        results = {
            'cutoff': Mock(success=True, data={'optimal_value': 120.0}),
            'kmesh': Mock(success=False, data={'error': 'Failed'}),
            'lattice': Mock(success=True, data={'a_opt': 5.653})
        }

        result = PipelineResult(results=results, config=get_fast_config())

        assert result.success is False  # No completamente exitoso
        assert result.total_stages == 3
        assert result.completed_stages == 2
        assert result.failed_stages == 1


class TestStageIntegration:
    """Tests de integración entre stages."""

    def test_cutoff_to_kmesh_data_flow(self, fast_config):
        """Test flujo de datos de cutoff a kmesh."""
        pipeline = PreconvergencePipeline(fast_config)

        # Simular resultado de cutoff
        cutoff_result = Mock()
        cutoff_result.success = True
        cutoff_result.data = {'optimal_value': 120.0}

        # Verificar que kmesh puede usar este dato
        kmesh_stage = pipeline.stages['kmesh']

        # Validar inputs
        previous_results = {'cutoff': cutoff_result}
        assert kmesh_stage.validate_inputs(previous_results)

    def test_kmesh_to_lattice_data_flow(self, fast_config):
        """Test flujo de datos de kmesh a lattice."""
        pipeline = PreconvergencePipeline(fast_config)

        # Simular resultados previos
        cutoff_result = Mock(success=True, data={'optimal_value': 120.0})
        kmesh_result = Mock(success=True, data={'optimal_kmesh': (4, 4, 4)})

        # Verificar que lattice puede usar estos datos
        lattice_stage = pipeline.stages['lattice']

        previous_results = {
            'cutoff': cutoff_result,
            'kmesh': kmesh_result
        }
        assert lattice_stage.validate_inputs(previous_results)