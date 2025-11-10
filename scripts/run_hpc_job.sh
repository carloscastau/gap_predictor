#!/bin/bash
# scripts/run_hpc_job.sh
# Script para ejecutar el pipeline en supercomputadoras con SLURM

#SBATCH --job-name=preconvergence-gaas
#SBATCH --output=preconv_%j.out
#SBATCH --error=preconv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=your_account

# Cargar módulos necesarios
module load python/3.11
module load openmpi/4.1.4
module load cuda/11.8  # Si GPUs disponibles

# Configurar variables de entorno
export PYSCF_MAX_MEMORY=65536  # 64GB
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1  # Si GPUs disponibles

# Crear directorio de trabajo
WORK_DIR="/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copiar archivos del proyecto
cp -r /path/to/your/project/* .

# Instalar dependencias si es necesario
pip install --user -r requirements.txt

# Ejecutar el pipeline
python scripts/run_preconvergence.py \
    --config config/hpc.yaml \
    --output_dir results \
    --verbose

# Copiar resultados de vuelta
cp -r results /path/to/your/project/results_${SLURM_JOB_ID}

echo "Job completed. Results copied to /path/to/your/project/results_${SLURM_JOB_ID}"