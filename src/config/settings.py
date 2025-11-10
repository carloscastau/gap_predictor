# src/config/settings.py
"""Configuración centralizada con validación."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import multiprocessing
import os


@dataclass
class PreconvergenceConfig:
    """Configuración centralizada con validación."""

    # Parámetros físicos
    lattice_constant: float = 5.653
    x_ga: float = 0.25
    sigma_ha: float = 0.01

    # Parámetros computacionales
    basis_set: str = "gth-dzvp"
    pseudopotential: str = "gth-pbe"
    xc_functional: str = "PBE"

    # Parámetros de convergencia
    cutoff_list: List[float] = field(default_factory=lambda: [80, 120, 160])
    kmesh_list: List[Tuple[int, int, int]] = field(default_factory=lambda: [(2,2,2), (4,4,4), (6,6,6)])

    # Configuración de paralelismo
    max_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() // 2))
    timeout_seconds: int = 300
    memory_limit_gb: float = 8.0

    # Configuración de salida
    output_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoint_interval: int = 60  # segundos

    # Configuración de stages
    stage_timeout: int = 600  # segundos por stage

    # Configuración de logging
    log_level: str = "INFO"
    log_file: str = "preconv.log"

    # Configuración de GPU (opcional)
    use_gpu: bool = False
    gpu_memory_limit: float = 4.0

    def __post_init__(self):
        """Validación automática de parámetros."""
        self._validate_parameters()
        self._setup_derived_config()

    def _validate_parameters(self):
        """Validación automática de parámetros."""
        # Validar parámetro de red
        if not (5.0 <= self.lattice_constant <= 6.5):
            raise ValueError(f"Parámetro de red {self.lattice_constant} fuera de rango físico (5.0-6.5 Å)")

        # Validar posición fraccionaria
        if not (0.2 <= self.x_ga <= 0.3):
            raise ValueError(f"Posición x_ga {self.x_ga} fuera de rango típico para zincblende (0.2-0.3)")

        # Validar listas de convergencia
        if not self.cutoff_list or len(self.cutoff_list) < 2:
            raise ValueError("cutoff_list debe contener al menos 2 valores")

        if not self.kmesh_list or len(self.kmesh_list) < 2:
            raise ValueError("kmesh_list debe contener al menos 2 mallas k-point")

        # Validar límites de memoria
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb debe ser positivo")

        # Validar timeouts
        if self.timeout_seconds <= 0 or self.stage_timeout <= 0:
            raise ValueError("Timeouts deben ser positivos")

    def _setup_derived_config(self):
        """Configurar parámetros derivados."""
        # Asegurar que el directorio de salida existe
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configurar variables de entorno para paralelismo
        if not self.use_gpu:
            os.environ.setdefault('OMP_NUM_THREADS', str(min(4, self.max_workers)))
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')

    def get_stage_config(self, stage_name: str) -> dict:
        """Obtener configuración específica para un stage."""
        base_config = {
            'timeout': self.stage_timeout,
            'memory_limit': self.memory_limit_gb,
            'checkpoint_interval': self.checkpoint_interval
        }

        # Configuraciones específicas por stage
        stage_configs = {
            'cutoff': {
                'early_stop_threshold': 1e-4,  # Ha
                'min_points': 3,
                'max_points': 10
            },
            'kmesh': {
                'early_stop_threshold': 1e-5,  # Ha
                'min_points': 3,
                'max_points': 8
            },
            'lattice': {
                'optimization_method': 'quadratic_fit',
                'tolerance': 1e-6,
                'max_iterations': 20
            },
            'bands': {
                'line_density': 50,
                'symmetry_tolerance': 1e-5
            },
            'slab': {
                'vacuum_size': 12.0,  # Å
                'miller_indices': [(0,0,1), (1,1,0)]
            }
        }

        return {**base_config, **stage_configs.get(stage_name, {})}

    def to_dict(self) -> dict:
        """Convertir configuración a diccionario."""
        return {
            'lattice_constant': self.lattice_constant,
            'x_ga': self.x_ga,
            'sigma_ha': self.sigma_ha,
            'basis_set': self.basis_set,
            'pseudopotential': self.pseudopotential,
            'xc_functional': self.xc_functional,
            'cutoff_list': self.cutoff_list,
            'kmesh_list': self.kmesh_list,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'memory_limit_gb': self.memory_limit_gb,
            'output_dir': str(self.output_dir),
            'checkpoint_interval': self.checkpoint_interval,
            'stage_timeout': self.stage_timeout,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'use_gpu': self.use_gpu,
            'gpu_memory_limit': self.gpu_memory_limit
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PreconvergenceConfig':
        """Crear configuración desde diccionario."""
        # Convertir output_dir de string a Path
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])

        return cls(**config_dict)

    def save_to_file(self, filepath: Path):
        """Guardar configuración a archivo YAML."""
        import yaml

        config_dict = self.to_dict()
        # Convertir Path a string para serialización
        config_dict['output_dir'] = str(config_dict['output_dir'])

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PreconvergenceConfig':
        """Cargar configuración desde archivo YAML."""
        import yaml

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)


# Configuraciones predefinidas para diferentes escenarios
def get_default_config() -> PreconvergenceConfig:
    """Configuración por defecto para desarrollo."""
    return PreconvergenceConfig()


def get_fast_config() -> PreconvergenceConfig:
    """Configuración rápida para pruebas."""
    return PreconvergenceConfig(
        cutoff_list=[80, 120],
        kmesh_list=[(2,2,2), (4,4,4)],
        timeout_seconds=60,
        stage_timeout=120,
        max_workers=2
    )


def get_hpc_config() -> PreconvergenceConfig:
    """Configuración optimizada para HPC."""
    return PreconvergenceConfig(
        max_workers=multiprocessing.cpu_count(),
        memory_limit_gb=32.0,
        timeout_seconds=600,
        stage_timeout=1800,
        use_gpu=True,
        gpu_memory_limit=8.0
    )


def get_production_config() -> PreconvergenceConfig:
    """Configuración de producción con alta precisión."""
    return PreconvergenceConfig(
        cutoff_list=[60, 80, 100, 120, 140, 160, 180],
        kmesh_list=[(2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (8,8,8)],
        timeout_seconds=900,
        stage_timeout=3600,
        memory_limit_gb=16.0,
        checkpoint_interval=30
    )