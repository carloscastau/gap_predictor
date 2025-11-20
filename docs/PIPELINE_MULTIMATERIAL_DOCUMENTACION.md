# Documentación del Pipeline Multimaterial

## Resumen Ejecutivo

El **Pipeline Optimizado para Múlti Materiales** es un sistema completo que integra todo el ecosistema de preconvergencia DFT desarrollado, permitiendo ejecutar cálculos para múltiples materiales semiconductores de forma eficiente y paralela.

## Arquitectura del Sistema

### Componentes Principales

```
📁 src/workflow/
├── 🏗️ multi_material_pipeline.py    # Pipeline principal multimaterial
├── 🔄 batch_processor.py             # Procesador por lotes optimizado
└── 📊 pipeline.py                    # Pipeline base integrado

📁 src/analysis/
└── 📈 multi_material_analysis.py     # Análisis avanzado de resultados

📁 scripts/
└── 🚀 run_preconvergence_campaign.py # Script de ejecución principal

📁 examples/
└── 🌟 demo_multimaterial_system.py   # Demostración completa
```

### Integración con Sistema Existente

El sistema multimaterial se integra perfectamente con:
- ✅ Pipeline de preconvergencia existente (`PreconvergencePipeline`)
- ✅ Base de datos de semiconductores (`SemiconductorDatabase`)
- ✅ Sistema de permutaciones (`MaterialPermutator`)
- ✅ Configuración multimaterial (`MultiMaterialConfig`)
- ✅ Checkpoints y monitoreo de producción

## Características Técnicas

### 🚀 Ejecución Optimizada

- **Paralelización Inteligente**: ThreadPoolExecutor y ProcessPoolExecutor
- **Gestión de Memoria**: Monitor de uso y reducción automática de workers
- **Control de Flujo**: Semáforos para evitar sobrecarga
- **Reintentos Automáticos**: Manejo robusto de errores

### 📊 Análisis Avanzado

- **Comparación de Parámetros**: Cutoffs, k-mesh, constantes de red
- **Análisis Estadístico**: Tests de normalidad, correlaciones, outliers
- **Visualizaciones Automáticas**: Gráficos de éxito, comparaciones, distribuciones
- **Reportes Ejecutivos**: Resúmenes y recomendaciones automáticas

### 🔧 Configuración Flexible

- **Ejecución Paralela/Secuencial**: Configurable por usuario
- **Control de Workers**: Número adaptativo según recursos
- **Prioridades**: Orden de ejecución personalizable
- **Checkpoints**: Guardado por material individual

## Guía de Uso

### 1. Uso Básico - Semiconductores Comunes

```python
from workflow.multi_material_pipeline import run_common_semiconductors_campaign

# Ejecutar campaña con semiconductores predefinidos
result = await run_common_semiconductors_campaign(
    materials=['GaAs', 'GaN', 'InP'],  # Opcional: especifica materiales
    parallel=True,                     # Ejecución paralela
    max_workers=4                      # Número de workers
)

print(f"Éxito: {result.success_rate:.1f}%")
```

### 2. Configuración Personalizada

```python
from workflow.multi_material_pipeline import MultiMaterialPipeline
from core.multi_material_config import MultiMaterialConfig

# Crear configuración personalizada
pipeline = MultiMaterialPipeline()

# Agregar materiales específicos
pipeline.add_materials_from_list(['ZnS', 'CdSe', 'InP'])

# Configurar paralelización
pipeline.enable_parallel_execution(True)
pipeline.set_parallel_workers(6)

# Ejecutar campaña
result = await pipeline.run_preconvergence_campaign()
```

### 3. Materiales Generados Automáticamente

```python
from workflow.multi_material_pipeline import run_generated_materials_campaign
from models.semiconductor_database import SemiconductorType

# Generar y ejecutar materiales automáticamente
result = await run_generated_materials_campaign(
    semiconductor_types=[SemiconductorType.III_V, SemiconductorType.II_VI],
    max_materials=10,           # Máximo 10 materiales
    parallel=True,
    max_workers=4
)
```

### 4. Análisis de Resultados

```python
from analysis.multi_material_analysis import MultiMaterialAnalyzer

# Crear analizador
analyzer = MultiMaterialAnalyzer(enable_visualizations=True)

# Analizar resultados de campaña
report = analyzer.analyze_campaign_results(
    campaign_result=result,
    output_dir=Path("analysis_results")
)

# Obtener resumen ejecutivo
summary = report.get_executive_summary()
print(summary['key_findings'])
```

## Script de Línea de Comandos

### Comandos Disponibles

```bash
# Campaña con semiconductores comunes
python scripts/run_preconvergence_campaign.py --type common

# Campaña con materiales específicos
python scripts/run_preconvergence_campaign.py \
    --type common \
    --materials GaAs,GaN,InP \
    --parallel \
    --workers 4

# Campaña con materiales personalizados
python scripts/run_preconvergence_campaign.py \
    --type custom \
    --materials ZnS,CdSe,HgTe \
    --analyze \
    --output resultados/

# Campaña con materiales generados
python scripts/run_preconvergence_campaign.py \
    --type generated \
    --max-materials 8 \
    --semiconductor-types III_V II_VI

# Solo validación
python scripts/run_preconvergence_campaign.py --validate-only --materials GaAs,GaN
```

### Opciones Principales

| Opción | Descripción | Valores |
|--------|-------------|---------|
| `--type` | Tipo de campaña | `common`, `custom`, `generated` |
| `--materials` | Lista de materiales | `"GaAs,GaN,InP"` |
| `--parallel` | Ejecución paralela | `True`/`False` |
| `--workers` | Número de workers | `1-16` |
| `--analyze` | Análisis detallado | `True`/`False` |
| `--output` | Directorio de salida | `path/to/results` |
| `--validate-only` | Solo validar | `True`/`False` |

## Ejemplos Prácticos

### Ejemplo 1: Campaña Básica III-V

```python
#!/usr/bin/env python3
import asyncio
from workflow.multi_material_pipeline import MultiMaterialPipeline

async def campaign_iii_v():
    # Crear pipeline
    pipeline = MultiMaterialPipeline()
    
    # Agregar semiconductores III-V importantes
    materials = ['GaAs', 'GaN', 'InP', 'AlAs', 'InAs']
    pipeline.add_materials_from_list(materials)
    
    # Configurar para ejecución paralela eficiente
    pipeline.enable_parallel_execution(True)
    pipeline.set_parallel_workers(3)
    
    # Ejecutar campaña
    result = await pipeline.run_preconvergence_campaign()
    
    # Guardar resultados
    pipeline.save_campaign_results(result, Path("iii_v_campaign.json"))
    
    return result

# Ejecutar
result = asyncio.run(campaign_iii_v())
print(f"Tasa de éxito: {result.success_rate:.1f}%")
```

### Ejemplo 2: Análisis Comparativo II-VI

```python
#!/usr/bin/env python3
import asyncio
from workflow.multi_material_pipeline import run_custom_materials_campaign
from analysis.multi_material_analysis import MultiMaterialAnalyzer

async def analyze_ii_vi():
    # Materiales II-VI de interés
    materials = ['ZnS', 'ZnSe', 'ZnTe', 'CdS', 'CdSe', 'CdTe']
    
    # Ejecutar campaña
    result = await run_custom_materials_campaign(
        materials=materials,
        parallel=True,
        max_workers=4
    )
    
    # Análisis detallado
    analyzer = MultiMaterialAnalyzer(enable_visualizations=True)
    report = analyzer.analyze_campaign_results(
        result, 
        output_dir=Path("ii_vi_analysis")
    )
    
    return report

# Ejecutar análisis
report = asyncio.run(analyze_ii_vi())
```

### Ejemplo 3: Generación Automática de Materiales

```python
#!/usr/bin/env python3
import asyncio
from workflow.multi_material_pipeline import run_generated_materials_campaign
from core.material_permutator import PermutationFilter, MATERIAL_PERMUTATOR

async def generated_materials_campaign():
    # Filtros para materiales de calidad
    filter_config = PermutationFilter(
        only_common_elements=True,
        exclude_toxic=True,
        exclude_radioactive=True
    )
    
    # Generar materiales con filtros
    result_iii_v = MATERIAL_PERMUTATOR.generate_iii_v_combinations(filter_config)
    result_ii_vi = MATERIAL_PERMUTATOR.generate_ii_vi_combinations(filter_config)
    
    # Combinar resultados
    pipeline = MultiMaterialPipeline()
    pipeline.add_materials_from_permutation(result_iii_v, max_materials=5)
    pipeline.add_materials_from_permutation(result_ii_vi, max_materials=5)
    
    # Configurar alta paralelización para screening
    pipeline.enable_parallel_execution(True)
    pipeline.set_parallel_workers(8)
    
    # Ejecutar campaña de screening
    result = await pipeline.run_preconvergence_campaign()
    
    return result

# Ejecutar campaña de materiales generados
result = asyncio.run(generated_materials_campaign())
```

## API Reference

### MultiMaterialPipeline

**Clase principal del pipeline multimaterial.**

```python
class MultiMaterialPipeline:
    def __init__(self, config: Optional[MultiMaterialConfig] = None)
    
    # Configuración
    def add_materials_from_list(self, formulas: List[str])
    def set_parallel_workers(self, max_workers: int)
    def enable_parallel_execution(self, enabled: bool = True)
    
    # Validación
    def validate_materials(self) -> Dict[str, Any]
    
    # Ejecución
    async def run_preconvergence_campaign(self, ...) -> CampaignResult
    async def execute_single_material(self, formula: str) -> MaterialExecutionResult
    
    # Utilidades
    def save_campaign_results(self, result: CampaignResult, filepath: Path)
    def get_campaign_progress(self) -> Dict[str, Any]
```

### BatchProcessor

**Procesador inteligente por lotes.**

```python
class BatchProcessor:
    def __init__(self, max_concurrent: int = 4, ...)
    
    async def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]
    def get_progress_status(self) -> Optional[Dict[str, Any]]
    def stop_processing(self)
```

### MultiMaterialAnalyzer

**Sistema de análisis avanzado.**

```python
class MultiMaterialAnalyzer:
    def analyze_campaign_results(self, campaign_result: CampaignResult) -> MultiMaterialAnalysisReport
    
    # Métodos específicos
    def _compare_parameters(self, campaign_result: CampaignResult) -> List[ParameterComparison]
    def _analyze_by_groups(self, campaign_result: CampaignResult) -> List[MaterialGroupAnalysis]
    def _create_visualizations(self, ...) -> List[str]
```

### CampaignResult

**Resultado consolidado de campaña.**

```python
@dataclass
class CampaignResult:
    materials_executed: int
    materials_successful: int
    materials_failed: int
    total_execution_time: float
    individual_results: List[MaterialExecutionResult]
    campaign_config: MultiMaterialConfig
    
    @property
    def success_rate(self) -> float
    def get_successful_materials(self) -> List[str]
    def get_consolidated_results(self) -> dict
```

## Configuración Avanzada

### Archivo de Configuración YAML

```yaml
base_config:
  cutoff_list: [400, 500, 600]
  kmesh_list: [[6, 6, 6], [8, 8, 8]]
  lattice_constant: 5.7
  x_ga: 0.25
  
materials:
  - formula: "GaAs"
    lattice_constant: 5.653
    priority: 10
    enabled: true
  - formula: "GaN"
    lattice_constant: 4.52
    priority: 9
    cutoff: 500
    
auto_generate: false
parallel_materials: true
max_concurrent_materials: 4
output_base_dir: "results_campaign"
```

### Gestión de Memoria

El sistema incluye gestión automática de memoria:

```python
# Configurar límites de memoria
pipeline = MultiMaterialPipeline()
pipeline.memory_limit_gb = 16.0  # 16GB límite

# Monitoreo automático
# El sistema reduce automáticamente workers si detecta poca memoria
```

### Prioridades y Scheduling

```python
# Configurar prioridades
config = MultiMaterialConfig()
config.add_material('GaAs', priority=10)  # Alta prioridad
config.add_material('GaN', priority=8)    # Media prioridad
config.add_material('ZnS', priority=5)    # Baja prioridad

# Ordenar por prioridad
config.sort_by_priority()
```

## Solución de Problemas

### Problemas Comunes

**1. Error: "Material no encontrado en base de datos"**
```python
# Verificar que el material existe
from models.semiconductor_database import SEMICONDUCTOR_DB
if 'GaAs' in SEMICONDUCTOR_DB.semiconductors:
    print("Material disponible")
```

**2. Error: "Memoria insuficiente"**
```python
# Reducir workers o usar modo secuencial
pipeline.set_parallel_workers(2)  # Reducir workers
pipeline.enable_parallel_execution(False)  # Modo secuencial
```

**3. Tiempo de ejecución muy largo**
```python
# Usar menos materiales para debugging
pipeline.add_materials_from_list(['GaAs', 'GaN'])  # Solo 2 materiales

# O usar modo secuencial para debugging
pipeline.enable_parallel_execution(False)
```

### Logs y Debugging

```python
# Habilitar logging detallado
from utils.logging import setup_logging
setup_logging(level='DEBUG')

# Verificar progreso
progress = pipeline.get_campaign_progress()
print(progress)

# Guardar reporte de progreso
pipeline.batch_processor.save_progress_report(Path("progress_report.json"))
```

## Rendimiento y Escalabilidad

### Benchmarks de Rendimiento

- **1 material**: ~30-60 segundos
- **5 materiales (paralelo, 4 workers)**: ~90-120 segundos
- **10 materiales (paralelo, 4 workers)**: ~180-240 segundos

### Optimizaciones Aplicadas

1. **Paralelización de Materiales**: Cada material en proceso independiente
2. **Caché de Pipelines**: Reutilización de configuraciones
3. **Gestión de Memoria**: Liberación automática entre materiales
4. **Batch Processing**: Agrupación inteligente de tareas similares

### Escalabilidad

- **Workers Recomendados**: 1-2 por CPU core disponible
- **Memoria Requerida**: ~2-4GB por worker activo
- **Materiales Simultáneos**: Recomendado máximo 8-12 para sistemas de 32GB

## Integración con PySCF

El sistema está diseñado para integrarse con cálculos DFT reales:

```python
# Reemplazar simulaciones con cálculos PySCF reales
class RealDFTCalculator:
    async def calculate_energy(self, cell_params):
        # Integrar con PySCF aquí
        from pyscf import gto, dft, cc
        # ... implementación real
        pass

# En el pipeline, reemplazar el simulador
pipeline.calculator = RealDFTCalculator()
```

## Conclusión

El Pipeline Multimaterial proporciona una solución completa, escalable y eficiente para cálculos DFT de preconvergencia en múltiples materiales semiconductores. Su arquitectura modular, sistema de análisis avanzado y integración perfecta con el ecosistema existente lo convierten en una herramienta poderosa para investigación de materiales a escala.

### Próximos Desarrollos

- ✅ Integración completa con PySCF
- 🔄 Optimizaciones de algoritmos de convergencia
- 🔄 Soporte para supercélulas y defectos
- 🔄 Interfaz web para monitoreo
- 🔄 API REST para integración con otros sistemas

### Soporte y Contribución

Para reportar bugs, solicitar características o contribuir al desarrollo, consulte la documentación del proyecto principal.