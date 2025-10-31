#!/bin/bash
#SBATCH --job-name=gaas_multi_node
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm-multi-%j.out
#SBATCH --error=slurm-multi-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Cargar módulos
module load singularity/3.8.0
module load python/3.10

# Configuración MPI (si es necesario)
# module load openmpi/4.1.0

# Variables de entorno
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=128000

# Directorio compartido
SHARED_DIR=/scratch/$USER/gaas_multi_$SLURM_JOB_ID
mkdir -p $SHARED_DIR

# Copiar archivos a directorio compartido
if [ $SLURM_PROCID -eq 0 ]; then
    cp -r $SLURM_SUBMIT_DIR/* $SHARED_DIR/
fi

# Sincronizar nodos
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c 'echo "Node $SLURMD_NODENAME ready"'

# Cambiar a directorio de trabajo
cd $SHARED_DIR

# Ejecutar pipeline distribuido
# Cada nodo procesa diferentes bases/moléculas
case $SLURM_PROCID in
    0)
        echo "Node 0: Processing basis set gth-szv"
        srun --exclusive --nodes=1 --ntasks=8 singularity exec \
            --bind $SHARED_DIR:/workspace \
            preconvergencia-gaas.sif \
            python preconvergencia_GaAs.py \
            --basis_list gth-szv \
            --cutoff_list 80,120,160,200 \
            --k_list 6x6x6,8x8x8,10x10x10 \
            --output_dir results_node0
        ;;
    1)
        echo "Node 1: Processing basis set gth-dzvp"
        srun --exclusive --nodes=1 --ntasks=8 singularity exec \
            --bind $SHARED_DIR:/workspace \
            preconvergencia-gaas.sif \
            python preconvergencia_GaAs.py \
            --basis_list gth-dzvp \
            --cutoff_list 100,150,200,250 \
            --k_list 8x8x8,10x10x10,12x12x12 \
            --output_dir results_node1
        ;;
    2)
        echo "Node 2: Processing basis set gth-tzvp"
        srun --exclusive --nodes=1 --ntasks=8 singularity exec \
            --bind $SHARED_DIR:/workspace \
            preconvergencia-gaas.sif \
            python advanced_optimization.py \
            --config_file optimization_config.json \
            --output_dir results_node2
        ;;
    3)
        echo "Node 3: Processing basis set gth-tzv2p"
        srun --exclusive --nodes=1 --ntasks=8 singularity exec \
            --bind $SHARED_DIR:/workspace \
            preconvergencia-gaas.sif \
            python incremental_pipeline.py \
            --reuse_previous on \
            --target_accuracy 1e-5 \
            --output_dir results_node3
        ;;
esac

# Recopilar resultados (solo en nodo maestro)
if [ $SLURM_PROCID -eq 0 ]; then
    echo "Consolidating results..."
    mkdir -p consolidated_results
    cp -r results_node* consolidated_results/
    cp -r consolidated_results $SLURM_SUBMIT_DIR/

    # Limpiar scratch
    rm -rf $SHARED_DIR
fi