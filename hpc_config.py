#!/usr/bin/env python3
"""
Configuración avanzada para entornos HPC con gestión inteligente de recursos.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class HPCConfig:
    """Configuración para entornos HPC."""

    # Configuración de paralelización
    omp_num_threads: int = 8
    openblas_num_threads: int = 1
    mkl_num_threads: int = 1
    pyscf_max_memory: int = 8000  # MB

    # Configuración de SLURM
    slurm_job_name: str = "gaas_preconvergence"
    slurm_partition: str = "gpu"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 8
    slurm_cpus_per_task: int = 4
    slurm_mem: str = "32GB"
    slurm_time: str = "24:00:00"

    # Configuración de Singularity
    singularity_image: str = "preconvergencia-gaas.sif"
    singularity_bind_paths: list = None

    # Configuración de tolerancias y timeouts
    convergence_timeout_s: int = 900
    memory_limit_mb: int = 32000
    checkpoint_interval_s: int = 300

    def __post_init__(self):
        if self.singularity_bind_paths is None:
            self.singularity_bind_paths = ["/scratch", "/tmp"]

    @classmethod
    def from_environment(cls) -> 'HPCConfig':
        """Crear configuración desde variables de entorno SLURM."""
        config = cls()

        # Detectar configuración SLURM
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            config.omp_num_threads = int(os.environ['SLURM_CPUS_PER_TASK'])
            config.slurm_cpus_per_task = config.omp_num_threads

        if 'SLURM_NNODES' in os.environ:
            config.slurm_nodes = int(os.environ['SLURM_NNODES'])

        if 'SLURM_NTASKS_PER_NODE' in os.environ:
            config.slurm_ntasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])

        if 'SLURM_JOB_NAME' in os.environ:
            config.slurm_job_name = os.environ['SLURM_JOB_NAME']

        if 'SLURM_JOB_PARTITION' in os.environ:
            config.slurm_partition = os.environ['SLURM_JOB_PARTITION']

        # Ajustar memoria basada en configuración
        total_memory = config.slurm_nodes * 32000  # MB por nodo
        config.pyscf_max_memory = min(total_memory, 128000)  # Máximo 128GB

        return config

    @classmethod
    def from_file(cls, config_file: Path) -> 'HPCConfig':
        """Cargar configuración desde archivo JSON."""
        if not config_file.exists():
            return cls()

        with open(config_file, 'r') as f:
            data = json.load(f)

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_file(self, config_file: Path):
        """Guardar configuración a archivo JSON."""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def apply_environment(self):
        """Aplicar configuración de entorno."""
        os.environ['OMP_NUM_THREADS'] = str(self.omp_num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.openblas_num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.mkl_num_threads)
        os.environ['PYSCF_MAX_MEMORY'] = str(self.pyscf_max_memory)

        print("=== CONFIGURACIÓN HPC APLICADA ===")
        print(f"OMP_NUM_THREADS: {self.omp_num_threads}")
        print(f"PYSCF_MAX_MEMORY: {self.pyscf_max_memory} MB")
        print(f"Nodos SLURM: {self.slurm_nodes}")
        print(f"Tareas por nodo: {self.slurm_ntasks_per_node}")
        print(f"CPUs por tarea: {self.slurm_cpus_per_task}")
        print("===================================")

    def generate_slurm_script(self, job_type: str = "single") -> str:
        """Generar script SLURM basado en configuración."""

        if job_type == "array":
            script = f"""#!/bin/bash
#SBATCH --job-name={self.slurm_job_name}_array
#SBATCH --partition={self.slurm_partition}
#SBATCH --nodes={self.slurm_nodes}
#SBATCH --ntasks-per-node={self.slurm_ntasks_per_node}
#SBATCH --cpus-per-task={self.slurm_cpus_per_task}
#SBATCH --mem={self.slurm_mem}
#SBATCH --time={self.slurm_time}
#SBATCH --output=slurm-array-%A-%a.out
#SBATCH --error=slurm-array-%A-%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL

# Configuración de entorno
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS={self.openblas_num_threads}
export MKL_NUM_THREADS={self.mkl_num_threads}
export PYSCF_MAX_MEMORY={self.pyscf_max_memory}

# Directorio de trabajo
WORKDIR=/scratch/$USER/{self.slurm_job_name}_$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID
mkdir -p $WORKDIR

# Copiar archivos
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Ejecutar trabajo específico del array
singularity exec --pwd /workspace \\
    --bind $WORKDIR:/workspace \\
    {self.singularity_image} \\
    python preconvergencia_GaAs.py \\
    --fast --timeout_s {self.convergence_timeout_s} \\
    --output_dir results_$SLURM_ARRAY_TASK_ID

# Copiar resultados
cp -r results_$SLURM_ARRAY_TASK_ID $SLURM_SUBMIT_DIR/

# Limpiar
rm -rf $WORKDIR
"""
        else:
            script = f"""#!/bin/bash
#SBATCH --job-name={self.slurm_job_name}
#SBATCH --partition={self.slurm_partition}
#SBATCH --nodes={self.slurm_nodes}
#SBATCH --ntasks-per-node={self.slurm_ntasks_per_node}
#SBATCH --cpus-per-task={self.slurm_cpus_per_task}
#SBATCH --mem={self.slurm_mem}
#SBATCH --time={self.slurm_time}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Configuración de entorno
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS={self.openblas_num_threads}
export MKL_NUM_THREADS={self.mkl_num_threads}
export PYSCF_MAX_MEMORY={self.pyscf_max_memory}

# Directorio de trabajo
WORKDIR=/scratch/$USER/{self.slurm_job_name}_$SLURM_JOB_ID
mkdir -p $WORKDIR

# Copiar archivos
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Ejecutar cálculo principal
singularity exec --pwd /workspace \\
    --bind $WORKDIR:/workspace \\
    {self.singularity_image} \\
    python incremental_pipeline.py \\
    --fast --reuse_previous on --target_accuracy 1e-4 \\
    --timeout_s {self.convergence_timeout_s}

# Copiar resultados
cp -r preconvergencia_out $SLURM_SUBMIT_DIR/

# Limpiar
rm -rf $WORKDIR
"""

        return script


def detect_hpc_environment() -> Dict[str, Any]:
    """Detectar características del entorno HPC."""
    env_info = {
        "is_slurm": "SLURM_JOB_ID" in os.environ,
        "is_singularity": "SINGULARITY_CONTAINER" in os.environ,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodes": int(os.environ.get("SLURM_NNODES", 1)),
        "slurm_cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        "available_memory_gb": None,
        "gpu_available": False
    }

    # Detectar memoria disponible
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    env_info["available_memory_gb"] = mem_kb / 1024 / 1024
                    break
    except:
        pass

    # Detectar GPU
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            env_info["gpu_available"] = True
            env_info["gpu_info"] = result.stdout.strip().split('\n')
    except:
        pass

    return env_info


if __name__ == "__main__":
    # Ejemplo de uso
    config = HPCConfig.from_environment()
    config.apply_environment()

    env_info = detect_hpc_environment()
    print("Información del entorno HPC:")
    print(json.dumps(env_info, indent=2))

    # Generar script SLURM
    slurm_script = config.generate_slurm_script("single")
    with open("generated_slurm_job.sh", "w") as f:
        f.write(slurm_script)

    print("Script SLURM generado: generated_slurm_job.sh")