# Guía de Despliegue en Supercomputo (HPC)

Esta guía proporciona instrucciones completas para desplegar y ejecutar cálculos DFT de preconvergencia GaAs en entornos de supercomputo usando SLURM y Singularity.

## 📋 Requisitos Previos

### En el Sistema Local
- Docker instalado y configurado
- Git para control de versiones
- Python 3.10+ con dependencias del proyecto

### En el Supercomputo
- SLURM como gestor de colas
- Singularity/Apptainer instalado
- Acceso a nodos de cómputo con GPUs (recomendado)
- Almacenamiento compartido (/scratch, /home)

## 🐳 Construcción del Contenedor Singularity

### 1. Construir Imagen Singularity

```bash
# En el sistema local o en el cluster (si permite Docker)
sudo docker build -t preconvergencia-gaas:latest .

# Convertir a Singularity (requiere Singularity instalado)
sudo docker run -d --name temp_container preconvergencia-gaas:latest tail -f /dev/null
sudo docker export temp_container | singularity build preconvergencia-gaas.sif docker-import://stdin
sudo docker rm temp_container
```

### 2. Construir Directamente con Singularity

```bash
# Copiar archivos del proyecto al cluster
scp -r . user@cluster:/path/to/project/

# En el cluster, construir la imagen
singularity build preconvergencia-gaas.sif Singularity.def
```

## 🔧 Configuración del Entorno HPC

### Variables de Entorno Recomendadas

```bash
# En ~/.bashrc o en el script de SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=32000  # MB por tarea
```

### Configuración Personalizada

```bash
# Crear archivo de configuración HPC
python hpc_config.py

# O configurar manualmente
cat > hpc_config.json << EOF
{
  "omp_num_threads": 8,
  "pyscf_max_memory": 32000,
  "slurm_partition": "gpu",
  "slurm_time": "24:00:00",
  "singularity_image": "preconvergencia-gaas.sif"
}
EOF
```

## 🚀 Ejecución en SLURM

### Trabajo Individual

```bash
# Ejecutar trabajo básico
sbatch slurm_job.sh

# Ver estado del trabajo
squeue -u $USER

# Ver salida del trabajo
tail -f slurm-*.out
```

### Trabajo de Array (Múltiples Configuraciones)

```bash
# Ejecutar array de trabajos
sbatch slurm_array_job.sh

# Monitorear progreso
squeue -u $USER --array
```

### Trabajo Multi-Nodo

```bash
# Para cálculos muy grandes
sbatch slurm_multi_node.sh
```

## 📊 Estrategias de Paralelización

### Nivel 1: Paralelización por Bases
- Cada nodo procesa diferentes bases GTH
- Ideal para barrido de bases
- Ejemplo: `slurm_array_job.sh`

### Nivel 2: Paralelización por Puntos de Energía
- Múltiples puntos de k-mesh o cutoff en paralelo
- Implementado en `incremental_pipeline.py`

### Nivel 3: Paralelización Interna de PySCF
- Paralelización automática por k-points
- Configurada vía `OMP_NUM_THREADS`

## 🔍 Monitoreo y Depuración

### Comandos Útiles de SLURM

```bash
# Ver colas disponibles
sinfo

# Ver trabajos en cola
squeue -p gpu

# Ver detalles de un trabajo
scontrol show job <job_id>

# Cancelar trabajo
scancel <job_id>
```

### Logs y Debugging

```bash
# Ver logs en tiempo real
tail -f preconvergencia_out/preconv.log

# Ver métricas de rendimiento
sacct -j <job_id> --format=JobID,JobName,Elapsed,CPUTime,MaxRSS

# Depurar problemas de memoria
sacct -j <job_id> --format=JobID,MaxRSS,MaxVMSize
```

## 📈 Optimización de Rendimiento

### Configuración por Tipo de Trabajo

| Tipo de Cálculo | CPUs por Tarea | Memoria (GB) | Tiempo Estimado |
|-----------------|----------------|--------------|-----------------|
| Preconvergencia rápida | 4 | 16 | 2-4 horas |
| Optimización completa | 8 | 32 | 8-24 horas |
| Barrido de bases | 8 | 64 | 24-48 horas |
| Cálculos de producción | 16 | 128 | 48-72 horas |

### Estrategias de Optimización

1. **Gestión de Memoria**:
   ```bash
   # Ajustar según recursos disponibles
   export PYSCF_MAX_MEMORY=$((SLURM_MEM_PER_NODE * 1000 * 8 / 10))  # 80% de RAM disponible
   ```

2. **Paralelización Inteligente**:
   ```bash
   # Para trabajos intensivos en CPU
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

   # Para trabajos con muchos k-points
   export OMP_NUM_THREADS=1  # Dejar que PySCF maneje la paralelización
   ```

3. **Uso de Scratch**:
   ```bash
   # Usar almacenamiento local rápido
   WORKDIR=/scratch/$USER/job_$SLURM_JOB_ID
   mkdir -p $WORKDIR
   cd $WORKDIR
   ```

## ✅ Validación de Reproducibilidad

### Verificación Automática

```bash
# Ejecutar validador después de cada cálculo
python reproducibility_validator.py

# Verificar integridad de archivos
python reproducibility_validator.py --output-file validation_report.json
```

### Métricas de Validación

- **Fingerprint del entorno**: Identificador único del setup
- **Convergencia de cutoff**: ΔE < 1 meV entre puntos
- **Convergencia de k-mesh**: ΔE < 0.1 meV entre mallas
- **Parámetro de red**: 5.5-5.8 Å (rango físico)
- **Gap electrónico**: 1.0-2.0 eV (GaAs típico)

## 🔄 Flujos de Trabajo Recomendados

### Para Desarrollo Local
```bash
# Validación rápida
python preconvergencia_GaAs.py --fast --timeout_s 300

# Con validación de reproducibilidad
python preconvergencia_GaAs.py --fast --make_report on
python reproducibility_validator.py
```

### Para Producción en HPC
```bash
# Construir contenedor
singularity build preconvergencia-gaas.sif Singularity.def

# Ejecutar trabajo optimizado
sbatch slurm_job.sh

# Validar resultados
python reproducibility_validator.py
```

### Para Estudios Paramétricos
```bash
# Barrido sistemático
sbatch slurm_array_job.sh

# Análisis de resultados
python diagnostics.py
```

## 🚨 Solución de Problemas

### Problemas Comunes

1. **Tiempo de espera agotado**:
   - Aumentar `--timeout_s`
   - Verificar recursos de SLURM
   - Considerar partición más rápida

2. **Error de memoria**:
   - Reducir `PYSCF_MAX_MEMORY`
   - Aumentar `SLURM_MEM`
   - Usar menos procesos en paralelo

3. **Problemas de convergencia**:
   - Revisar parámetros iniciales
   - Ajustar `sigma_ha`
   - Verificar estructura cristalina

4. **Problemas de Singularity**:
   - Verificar versión de Singularity
   - Comprobar permisos de archivos
   - Revisar bind mounts

### Logs de Diagnóstico

```bash
# Ver logs detallados
tail -f preconvergencia_out/preconv.log

# Ver métricas de SLURM
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed,CPUTime,MaxRSS

# Depurar contenedor
singularity shell preconvergencia-gaas.sif
```

## 📚 Referencias y Recursos

- [Documentación SLURM](https://slurm.schedmd.com/documentation.html)
- [Documentación Singularity](https://sylabs.io/docs/)
- [PySCF Documentation](https://pyscf.org/)
- [Guía de Optimización](./optimization_guide.md)

## 🤝 Soporte

Para problemas específicos del cluster:
1. Consultar documentación local del HPC
2. Contactar administrador del sistema
3. Revisar logs detallados del trabajo
4. Usar herramientas de diagnóstico incluidas

---

**Nota**: Esta guía está optimizada para clusters con SLURM y Singularity. Adaptar según la configuración específica del supercomputo utilizado.