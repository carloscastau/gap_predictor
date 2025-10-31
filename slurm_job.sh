#!/bin/bash
#SBATCH --job-name=gaas_preconvergence
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Cargar módulos necesarios (ajustar según el cluster)
module load singularity/3.8.0
module load python/3.10

# Configuración de variables de entorno para optimización
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=32000  # MB

# Directorio de trabajo
WORKDIR=/scratch/$USER/gaas_preconvergence_$SLURM_JOB_ID
mkdir -p $WORKDIR

# Copiar archivos necesarios
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Ejecutar contenedor Singularity
singularity exec --pwd /workspace \
    --bind $WORKDIR:/workspace \
    preconvergencia-gaas.sif \
    python incremental_pipeline.py \
    --fast \
    --reuse_previous on \
    --target_accuracy 1e-4 \
    --timeout_s 3600

# Copiar resultados de vuelta
cp -r preconvergencia_out $SLURM_SUBMIT_DIR/

# Limpiar scratch
rm -rf $WORKDIR