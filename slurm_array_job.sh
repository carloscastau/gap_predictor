#!/bin/bash
#SBATCH --job-name=gaas_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm-array-%A-%a.out
#SBATCH --error=slurm-array-%A-%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Cargar módulos
module load singularity/3.8.0
module load python/3.10

# Configuración de paralelización
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=64000

# Directorio de trabajo
WORKDIR=/scratch/$USER/gaas_array_$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID
mkdir -p $WORKDIR

# Copiar archivos
cp -r $SLURM_SUBMIT_DIR/* $WORKDIR/
cd $WORKDIR

# Definir configuraciones para cada tarea del array
declare -a BASIS_SETS=("gth-szv" "gth-dzvp" "gth-tzvp" "gth-tzv2p")
declare -a CUTOFF_LISTS=("80,120,160" "100,150,200" "120,180,240" "150,200,250")
declare -a KMESH_LISTS=("4x4x4,6x6x6" "6x6x6,8x8x8" "8x8x8,10x10x10" "10x10x10,12x12x12")

# Calcular índices para distribución de tareas
BASIS_IDX=$((SLURM_ARRAY_TASK_ID / 3))
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % 3))

BASIS=${BASIS_SETS[$BASIS_IDX]}
CUTOFF_LIST=${CUTOFF_LISTS[$CONFIG_IDX]}
KMESH_LIST=${KMESH_LISTS[$CONFIG_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: Basis=$BASIS, Cutoff=$CUTOFF_LIST, Kmesh=$KMESH_LIST"

# Ejecutar cálculo específico
singularity exec --pwd /workspace \
    --bind $WORKDIR:/workspace \
    preconvergencia-gaas.sif \
    python preconvergencia_GaAs.py \
    --basis_list $BASIS \
    --cutoff_list $CUTOFF_LIST \
    --k_list $KMESH_LIST \
    --a0 5.653 --da 0.02 --npoints_side 3 \
    --sigma_ha 0.01 --xc PBE \
    --timeout_s 1800 --dos on --make_report on \
    --output_dir preconvergencia_out_task_$SLURM_ARRAY_TASK_ID

# Copiar resultados
cp -r preconvergencia_out_task_$SLURM_ARRAY_TASK_ID $SLURM_SUBMIT_DIR/

# Limpiar
rm -rf $WORKDIR