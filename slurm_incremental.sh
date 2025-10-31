#!/bin/bash
#SBATCH --job-name=gaas_incremental
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm-incremental-%j.out
#SBATCH --error=slurm-incremental-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Cargar módulos necesarios
module load singularity/3.8.0
module load python/3.10

# Configuración optimizada para cálculos incrementales
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=64000  # MB

# Directorio de trabajo persistente para reutilización de datos
WORKDIR=/scratch/$USER/gaas_incremental_$SLURM_JOB_ID
mkdir -p $WORKDIR

# Copiar archivos y datos previos
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Ejecutar pipeline incremental optimizado
singularity exec --pwd /workspace \
    --bind $WORKDIR:/workspace \
    preconvergencia-gaas.sif \
    python incremental_pipeline.py \
    --fast \
    --reuse_previous on \
    --target_accuracy 1e-4 \
    --timeout_s 7200 \
    --hpc_mode single

# Copiar resultados de vuelta
cp -r preconvergencia_out $SLURM_SUBMIT_DIR/
cp -r incremental_summary_*.json $SLURM_SUBMIT_DIR/

# Limpiar pero mantener datos para reutilización futura
# rm -rf $WORKDIR  # Comentar para debugging