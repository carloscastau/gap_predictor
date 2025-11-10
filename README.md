# Preconvergencia DFT/PBC para GaAs - Versión Refactorizada

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/PySCF-2.3.0-green.svg)](https://pyscf.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pipeline modular y escalable para preconvergencia DFT/PBC optimizado para GaAs, con soporte completo para ejecución en entornos locales, Docker y supercomputadoras.

## 🚀 Características Principales

- **Arquitectura Modular**: Código organizado en módulos independientes con responsabilidades claras
- **Configuración Flexible**: Soporte para múltiples entornos (local, Docker, HPC)
- **Paralelización Inteligente**: Optimización automática de recursos computacionales
- **Sistema de Stages**: Pipeline con stages modulares e independientes
- **Checkpoints Automáticos**: Recuperación automática de fallos y reanudación
- **Logging Estructurado**: Monitoreo completo del rendimiento y diagnóstico
- **Tests Completos**: Cobertura unitaria e integración

## 📋 Requisitos

- Python 3.9+
- PySCF 2.3.0
- NumPy, SciPy, Pandas, Matplotlib
- PyMatGen, SPGLIB

## 🛠️ Instalación

### Opción 1: Instalación Directa
```bash
git clone <repository-url>
cd preconvergencia-gaas
pip install -r requirements.txt
pip install -e .
```

### Opción 2: Docker
```bash
# Construir imagen
docker build -t preconvergence-gaas .

# Ejecutar contenedor
docker run -v $(pwd)/results:/app/results preconvergence-gaas
```

### Opción 3: Supercomputadora (HPC)
```bash
# Configurar módulos específicos de tu cluster
module load python/3.11 openmpi/4.1.4 cuda/11.8

# Instalar dependencias
pip install --user -r requirements.txt

# Ejecutar con configuración HPC
python scripts/run_preconvergence.py --config config/hpc.yaml
```

## 🎯 Uso

### Ejecución Local Rápida
```bash
# Configuración rápida para pruebas
python scripts/run_preconvergence.py --fast
```

### Ejecución con Configuración Personalizada
```bash
# Usar configuración específica
python scripts/run_preconvergence.py --config config/production.yaml

# Especificar directorio de salida
python scripts/run_preconvergence.py --output_dir my_results
```

### Reanudar desde Checkpoint
```bash
# Continuar desde un checkpoint anterior
python scripts/run_preconvergence.py --resume checkpoint_name
```

### Docker
```bash
# Ejecutar en contenedor con configuración optimizada
docker run -v $(pwd)/results:/app/results preconvergence-gaas \
    --config config/docker.yaml
```

### Supercomputadora (SLURM)
```bash
# Enviar job a cola SLURM
sbatch scripts/run_hpc_job.sh

# O ejecutar directamente
srun python scripts/run_preconvergence.py --config config/hpc.yaml
```

## ⚙️ Configuración

### Archivos de Configuración Disponibles

- **`config/default.yaml`**: Configuración estándar
- **`config/docker.yaml`**: Optimizada para contenedores Docker
- **`config/hpc.yaml`**: Optimizada para supercomputadoras
- **`config/fast.yaml`**: Configuración rápida para pruebas

### Parámetros Principales

```yaml
# Parámetros físicos
lattice_constant: 5.653  # Parámetro de red (Å)
x_ga: 0.25              # Posición Ga en (x,x,x)
sigma_ha: 0.01          # Smearing Fermi-Dirac (Ha)

# Parámetros computacionales
basis_set: "gth-dzvp"           # Base GTH
pseudopotential: "gth-pbe"      # Pseudopotencial
xc_functional: "PBE"            # Funcional de intercambio-correlación

# Convergencia
cutoff_list: [80, 120, 160]     # Cutoffs de plano de ondas (Ry)
kmesh_list: [[2,2,2], [4,4,4]]  # Mallas k-point

# Paralelización
max_workers: 4                  # Número máximo de workers
timeout_seconds: 300            # Timeout por cálculo (s)
memory_limit_gb: 8.0           # Límite de memoria (GB)
```

## 📊 Resultados

El pipeline genera automáticamente:

- **Gráficas de convergencia** para cutoff, k-mesh y parámetro de red
- **Estructura de bandas** y densidad de estados
- **Reportes HTML** con análisis completo
- **Archivos CSV** con datos numéricos
- **Logs estructurados** con métricas de rendimiento

### Estructura de Salida
```
results/
├── cutoff/
│   ├── cutoff.csv
│   └── E_vs_cutoff.png
├── kmesh/
│   ├── kmesh.csv
│   └── E_vs_kmesh.png
├── lattice/
│   ├── lattice_optimization.csv
│   └── advanced_optimization.png
├── bands/
│   ├── bands.csv
│   ├── bands.png
│   └── gap_summary.csv
├── checkpoints/
│   └── checkpoint_*.json
├── logs/
│   └── preconv.log
└── visualization_report/
    ├── convergence_overview.png
    ├── computational_efficiency.png
    └── preconvergence_report.html
```

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest

# Tests con cobertura
pytest --cov=src --cov-report=html

# Tests específicos
pytest tests/unit/test_config.py
pytest tests/integration/test_pipeline.py
```

## 🏗️ Arquitectura

```
preconvergencia-gaas/
├── src/
│   ├── config/          # Configuración centralizada
│   ├── core/            # Componentes principales (DFT, paralelización)
│   ├── models/          # Modelos de datos
│   ├── workflow/        # Pipeline y stages
│   │   ├── stages/      # Stages individuales
│   │   └── checkpoint/  # Sistema de checkpoints
│   ├── analysis/        # Análisis estadístico
│   ├── visualization/   # Generadores de gráficos
│   └── utils/           # Utilidades (logging, etc.)
├── tests/               # Tests unitarios e integración
├── scripts/             # Scripts de ejecución
├── config/              # Archivos de configuración YAML
└── docs/                # Documentación
```

## 🔧 Desarrollo

### Añadir Nuevo Stage
```python
# src/workflow/stages/new_stage.py
from .base import PipelineStage

class NewStage(PipelineStage):
    def get_dependencies(self) -> List[str]:
        return ["previous_stage"]

    async def execute(self, previous_results: Dict[str, StageResult]) -> StageResult:
        # Implementación del stage
        pass
```

### Añadir Nueva Configuración
```python
# src/config/settings.py
def get_custom_config() -> PreconvergenceConfig:
    return PreconvergenceConfig(
        # Parámetros personalizados
        cutoff_list=[100, 150, 200],
        max_workers=8,
        # ...
    )
```

## 📈 Rendimiento

### Benchmarks Típicos

| Configuración | Tiempo Estimado | Memoria | CPUs |
|---------------|----------------|---------|------|
| `fast` | 5-15 min | 2-4 GB | 1-2 |
| `default` | 30-60 min | 4-8 GB | 2-4 |
| `production` | 2-6 horas | 8-16 GB | 4-8 |
| `hpc` | 1-4 horas | 32-128 GB | 16+ |

### Optimizaciones Implementadas

- **Paralelización por tareas**: Cada punto de cálculo independiente se ejecuta en paralelo
- **Agrupamiento inteligente**: Tareas similares se ejecutan juntas para optimizar caché
- **Control de flujo**: Limitación de concurrencia para evitar sobrecarga de memoria
- **Early stopping**: Detención anticipada basada en criterios de convergencia
- **Checkpoints incrementales**: Guardado periódico del progreso

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- PySCF por el framework DFT
- PyMatGen por herramientas de cristalografía
- Comunidad científica de Python por las mejores prácticas

## 📞 Soporte

Para soporte técnico o preguntas:

1. Revisa la documentación en `docs/`
2. Abre un issue en GitHub
3. Contacta al equipo de desarrollo

---

**Nota**: Este proyecto está diseñado siguiendo las mejores prácticas de computación científica con Python, sirviendo como base sólida para proyectos similares en física computacional y química cuántica.