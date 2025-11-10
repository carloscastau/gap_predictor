# tests/unit/test_config.py
"""Tests para configuración."""

import pytest
from pathlib import Path

from src.config.settings import PreconvergenceConfig, get_default_config, get_fast_config


class TestPreconvergenceConfig:
    """Tests para PreconvergenceConfig."""

    def test_default_config_creation(self):
        """Test creación de configuración por defecto."""
        config = PreconvergenceConfig()

        assert config.lattice_constant == 5.653
        assert config.x_ga == 0.25
        assert config.basis_set == "gth-dzvp"
        assert config.cutoff_list == [80, 120, 160]
        assert config.kmesh_list == [(2, 2, 2), (4, 4, 4), (6, 6, 6)]

    def test_config_validation(self):
        """Test validación de configuración."""
        # Configuración válida
        config = PreconvergenceConfig(lattice_constant=5.5, x_ga=0.25)
        assert config.lattice_constant == 5.5

        # Configuración inválida - parámetro de red demasiado pequeño
        with pytest.raises(ValueError, match="Parámetro de red.*fuera de rango"):
            PreconvergenceConfig(lattice_constant=4.0)

        # Configuración inválida - posición x_ga fuera de rango
        with pytest.raises(ValueError, match="Posición x_ga.*fuera de rango"):
            PreconvergenceConfig(x_ga=0.1)

    def test_config_to_dict(self):
        """Test conversión a diccionario."""
        config = PreconvergenceConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['lattice_constant'] == 5.653
        assert config_dict['basis_set'] == "gth-dzvp"

    def test_config_from_dict(self):
        """Test creación desde diccionario."""
        config_dict = {
            'lattice_constant': 5.6,
            'x_ga': 0.25,
            'basis_set': 'gth-tzvp',
            'cutoff_list': [100, 140],
            'kmesh_list': [(3, 3, 3), (5, 5, 5)]
        }

        config = PreconvergenceConfig.from_dict(config_dict)

        assert config.lattice_constant == 5.6
        assert config.basis_set == 'gth-tzvp'
        assert config.cutoff_list == [100, 140]

    def test_get_stage_config(self):
        """Test obtención de configuración por stage."""
        config = PreconvergenceConfig()

        cutoff_config = config.get_stage_config('cutoff')
        assert 'timeout' in cutoff_config
        assert 'early_stop_threshold' in cutoff_config

        kmesh_config = config.get_stage_config('kmesh')
        assert 'timeout' in kmesh_config
        assert 'early_stop_threshold' in kmesh_config

    def test_fast_config(self):
        """Test configuración rápida."""
        config = get_fast_config()

        assert len(config.cutoff_list) < len(get_default_config().cutoff_list)
        assert config.timeout_seconds < get_default_config().timeout_seconds
        assert config.max_workers == 2

    def test_config_file_operations(self, tmp_path):
        """Test operaciones con archivos de configuración."""
        config = PreconvergenceConfig(lattice_constant=5.6, basis_set='gth-tzvp')

        # Guardar configuración
        config_file = tmp_path / "test_config.yaml"
        config.save_to_file(config_file)

        assert config_file.exists()

        # Cargar configuración
        loaded_config = PreconvergenceConfig.load_from_file(config_file)

        assert loaded_config.lattice_constant == 5.6
        assert loaded_config.basis_set == 'gth-tzvp'