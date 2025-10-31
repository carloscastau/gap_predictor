#!/bin/bash
#SBATCH --job-name=gaas_recovery
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm-recovery-%j.out
#SBATCH --error=slurm-recovery-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Cargar módulos necesarios
module load singularity/3.8.0
module load python/3.10

# Configuración para recuperación
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=32000  # MB

# Directorio de trabajo
WORKDIR=/scratch/$USER/gaas_recovery_$SLURM_JOB_ID
mkdir -p $WORKDIR

# Copiar archivos y datos previos
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Verificar checkpoints disponibles
echo "Verificando checkpoints disponibles..."
python resume_checkpoint.py preconvergencia_out

# Intentar recuperación automática
echo "Intentando recuperación automática..."
singularity exec --pwd /workspace \
    --bind $WORKDIR:/workspace \
    preconvergencia-gaas.sif \
    python preconvergencia_GaAs.py \
    --resume on \
    --fast \
    --timeout_s 1800

# Si la recuperación automática falla, usar pipeline incremental
if [ $? -ne 0 ]; then
    echo "Recuperación automática falló, intentando pipeline incremental..."
    singularity exec --pwd /workspace \
        --bind $WORKDIR:/workspace \
        preconvergencia-gaas.sif \
        python incremental_pipeline.py \
        --fast \
        --reuse_previous on \
        --target_accuracy 1e-4 \
        --timeout_s 3600
fi

# Copiar resultados de vuelta
cp -r preconvergencia_out $SLURM_SUBMIT_DIR/

# Limpiar scratch
rm -rf $WORKDIR