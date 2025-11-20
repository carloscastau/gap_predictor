# Guía de Usuario Integral - Sistema de Preconvergencia Multimaterial

## 🎯 Bienvenido al Sistema de Preconvergencia DFT Multimaterial

Esta guía integral te llevará desde los conceptos básicos hasta casos de uso avanzados del sistema de preconvergencia multimaterial para semiconductores III-V y II-VI. Diseñada tanto para **nuevos usuarios** como para **investigadores experimentados**.

---

## 📚 Índice de la Guía

1. [**Introducción y Conceptos Básicos**](#1-introducción-y-conceptos-básicos)
2. [**Configuración Inicial**](#2-configuración-inicial)
3. [**Tutorial Paso a Paso para Nuevos Usuarios**](#3-tutorial-paso-a-paso-para-nuevos-usuarios)
4. [**Casos de Uso Específicos**](#4-casos-de-uso-específicos)
5. [**Mejores Prácticas y Recomendaciones**](#5-mejores-prácticas-y-recomendaciones)
6. [**Troubleshooting Común**](#6-troubleshooting-común)
7. [**Preguntas Frecuentes (FAQ)**](#7-preguntas-frecuentes-faq)

---

## 1. Introducción y Conceptos Básicos

### ¿Qué es la Preconvergencia DFT?

La **preconvergencia DFT** (Density Functional Theory) es un proceso crítico para establecer parámetros computacionales óptimos antes de realizar cálculos de estructura electrónica en materiales. Los parámetros principales incluyen:

- **Cutoff Energy**: Energía de corte para funciones de onda
- **K-mesh**: Malla de puntos k en la zona de Brillouin
- **Lattice Constant**: Constante de red cristalina optimizada

### ¿Qué hace especial a este sistema?

🎯 **Sistema Multimaterial**: Procesa múltiples semiconductores simultáneamente  
🔄 **Paralelización Inteligente**: Optimiza el uso de recursos computacionales  
🤖 **Generación Automática**: Crea combinaciones III-V y II-VI automáticamente  
📊 **Análisis Avanzado**: Compara y analiza resultados estadísticamente  
🛡️ **Sistema Robusto**: Recuperación automática de errores y checkpoints

### ¿Para quién es este sistema?

| Tipo de Usuario | Nivel | Casos de Uso Principales |
|------------------|-------|--------------------------|
| **Investigador Novato** | Principiante | Validación de un material, aprendizaje |
| **Investigador Senior** | Intermedio | Estudios comparativos, screening |
| **Ingeniero de Producción** | Avanzado | Optimización masiva, reportes |
| **Administrador de Sistema** | Técnico | Configuración, mantenimiento |

---

## 2. Configuración Inicial

### 2.1 Requisitos del Sistema

#### Requisitos Mínimos
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB libres
- **OS**: Linux, macOS, Windows (WSL2)

#### Requisitos Recomendados
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: SSD con 50+ GB libres
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+)

### 2.2 Instalación Rápida

#### Opción A: Instalación Directa (Recomendada para desarrollo)
```bash
# Clonar repositorio
git clone <repository-url>
cd preconvergencia-gaas

# Crear entorno virtual
python -m venv venv_preconvergencia
source venv_preconvergencia/bin/activate  # Linux/macOS
# o venv_preconvergencia\Scripts\activate  # Windows

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

#### Opción B: Docker (Recomendada para producción)
```bash
# Construir imagen
docker build -t preconvergence-multimaterial .

# Ejecutar contenedor con directorio persistente
docker run -it --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config:/app/config \
  preconvergence-multimaterial
```

### 2.3 Verificación de Instalación

```bash
# Verificar que todo funciona
python examples/uso_basico_multimaterial.py --quick

# Si ves "✅ Sistema funcionando correctamente", ¡estás listo!
```

### 2.4 Configuración del Entorno

#### Variables de Entorno (Opcional)
```bash
# Agregar a ~/.bashrc o ~/.zshrc
export PRECONV_ROOT="/path/to/preconvergencia-gaas"
export PRECONV_RESULTS="$PRECONV_ROOT/results"
export PYTHONPATH="$PRECONV_ROOT:$PYTHONPATH"
```

#### Configuración de Logging
```python
# config/logging.yaml (opcional)
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
root:
  level: INFO
  handlers: [console]
```

---

## 3. Tutorial Paso a Paso para Nuevos Usuarios

### 🎓 Tutorial 1: Tu Primer Análisis (15 minutos)

**Objetivo**: Ejecutar preconvergencia para GaAs y entender los resultados.

#### Paso 1: Preparación (2 minutos)
```python
# Crear archivo: mi_primer_analisis.py
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("🚀 Iniciando tu primer análisis...")
```

#### Paso 2: Análisis Simple (10 minutos)
```python
import asyncio
from workflow.pipeline import PreconvergencePipeline
from config.settings import get_fast_config

async def mi_primer_analisis():
    # 1. Configuración básica
    print("⚙️  Configurando análisis...")
    config = get_fast_config()
    
    # 2. Cambiar material a GaAs
    config.lattice_constant = 5.653  # Å - valor experimental
    
    # 3. Crear pipeline
    pipeline = PreconvergencePipeline(config)
    
    # 4. Ejecutar
    print("🔬 Ejecutando análisis de GaAs...")
    print("   Esto tomará 5-10 minutos...")
    
    result = await pipeline.execute()
    
    # 5. Mostrar resultados
    if result.success:
        print("✅ ¡Análisis exitoso!")
        print(f"   • Cutoff óptimo: {result.optimal_parameters.cutoff} Ry")
        print(f"   • K-mesh óptimo: {result.optimal_parameters.kmesh}")
        print(f"   • Lattice optimizado: {result.optimal_parameters.lattice_constant:.4f} Å")
    else:
        print(f"❌ Error: {result.error_message}")
    
    return result

# Ejecutar análisis
result = asyncio.run(mi_primer_analisis())
```

#### Paso 3: Ejecutar (3 minutos)
```bash
python mi_primer_analisis.py
```

**¿Qué deberías ver?**
```
🚀 Iniciando tu primer análisis...
⚙️  Configurando análisis...
🔬 Ejecutando análisis de GaAs...
   Esto tomará 5-10 minutos...
[Barras de progreso...]
✅ ¡Análisis exitoso!
   • Cutoff óptimo: 500 Ry
   • K-mesh óptimo: (6, 6, 6)
   • Lattice optimizado: 5.6528 Å
```

### 🎓 Tutorial 2: Análisis Multimaterial Básico (20 minutos)

**Objetivo**: Procesar 3 semiconductores simultáneamente.

```python
# Crear archivo: analisis_multimaterial.py
import asyncio
from workflow.multi_material_pipeline import run_common_semiconductors_campaign

async def analisis_basico_multimaterial():
    print("🎯 Análisis Multimaterial Básico")
    print("=" * 40)
    
    # Seleccionar materiales
    materiales = ['GaAs', 'GaN', 'InP']
    
    print(f"📋 Materiales: {', '.join(materiales)}")
    print(f"⚙️  Paralelización: 3 workers")
    
    # Ejecutar campaña
    result = await run_common_semiconductors_campaign(
        materials=materiales,
        parallel=True,
        max_workers=3
    )
    
    # Mostrar resultados
    print(f"\n📊 Resultados:")
    print(f"   • Procesados: {result.materials_executed}")
    print(f"   • Exitosos: {result.materials_successful}")
    print(f"   • Tasa de éxito: {result.success_rate:.1f}%")
    print(f"   • Tiempo total: {result.total_execution_time/60:.1f} min")
    
    return result

result = asyncio.run(analisis_basico_multimaterial())
```

### 🎓 Tutorial 3: Generación de Materiales (10 minutos)

**Objetivo**: Generar combinaciones automáticas de semiconductores.

```python
# Crear archivo: generar_materiales.py
from core.material_permutator import generate_all_iii_v, generate_all_ii_vi, PermutationFilter

def generar_materiales():
    print("🧪 Generación Automática de Materiales")
    print("=" * 45)
    
    # Configurar filtros
    filtros = PermutationFilter(
        only_common_elements=True,
        exclude_toxic=True
    )
    
    # Generar III-V
    print("🔄 Generando semiconductores III-V...")
    resultado_iii_v = generate_all_iii_v(filtros)
    
    print(f"   • Generadas: {resultado_iii_v.total_generated}")
    print(f"   • Aceptadas: {resultado_iii_v.total_accepted}")
    print(f"   • Tasa: {resultado_iii_v.acceptance_rate:.1f}%")
    
    # Mostrar ejemplos
    print(f"\n📋 Ejemplos de materiales generados:")
    for i, semiconductor in enumerate(resultado_iii_v.filtered_combinations[:5]):
        lattice = semiconductor.estimate_lattice_constant()
        print(f"   {i+1}. {semiconductor.formula}: a≈{lattice:.3f}Å")
    
    return resultado_iii_v

resultados = generar_materiales()
```

---

## 4. Casos de Uso Específicos

### 🔬 Caso de Uso 1: Investigación Científica

**Escenario**: Comparación sistemática de propiedades electrónicas en semiconductores III-V.

#### Objetivo
Estudiar tendencias en constantes de red y band gaps para publicar resultados.

#### Implementación
```python
# caso_investigacion.py
import asyncio
from workflow.multi_material_pipeline import run_custom_materials_campaign
from analysis.multi_material_analysis import MultiMaterialAnalyzer

async def estudio_iii_v_vs_ii_vi():
    print("📊 Estudio: III-V vs II-VI")
    print("=" * 35)
    
    # Materiales de estudio
    materiales_iii_v = ['GaAs', 'GaN', 'InP', 'AlAs', 'InAs']
    materiales_ii_vi = ['ZnS', 'ZnSe', 'ZnTe', 'CdS', 'CdSe']
    
    # Ejecutar estudios por separado
    print("🔬 Procesando III-V...")
    resultado_iii_v = await run_custom_materials_campaign(
        materials=materiales_iii_v,
        parallel=True,
        max_workers=5
    )
    
    print("⚗️  Procesando II-VI...")
    resultado_ii_vi = await run_custom_materials_campaign(
        materials=materiales_ii_vi,
        parallel=True,
        max_workers=5
    )
    
    # Análisis comparativo
    print("\n📈 Análisis Comparativo...")
    analyzer = MultiMaterialAnalyzer()
    
    # Comparar estadísticas
    print(f"III-V:")
    print(f"   • Tasa de éxito: {resultado_iii_v.success_rate:.1f}%")
    print(f"   • Tiempo promedio: {resultado_iii_v.average_execution_time:.1f}s")
    
    print(f"II-VI:")
    print(f"   • Tasa de éxito: {resultado_ii_vi.success_rate:.1f}%")
    print(f"   • Tiempo promedio: {resultado_ii_vi.average_execution_time:.1f}s")
    
    return resultado_iii_v, resultado_ii_vi

# Ejecutar estudio
resultados_iii_v, resultados_ii_vi = asyncio.run(estudio_iii_v_vs_ii_vi())
```

### 🏭 Caso de Uso 2: Producción Industrial

**Escenario**: Screening masivo de materiales para fabricación de dispositivos LED.

#### Objetivo
Identificar semiconductores óptimos para LED azul (band gap 2.5-3.5 eV).

```python
# caso_produccion.py
from core.material_permutator import MATERIAL_PERMUTATOR, PermutationFilter

def screening_led_azul():
    print("💡 Screening para LED Azul")
    print("=" * 30)
    
    # Filtros específicos para LED
    filtros_led = PermutationFilter(
        only_common_elements=True,
        exclude_toxic=True,
        max_ionic_radius_ratio=2.0,
        min_electronegativity_diff=0.5
    )
    
    # Generar candidatos
    candidatos = []
    
    for sem_type in ['III_V', 'II_VI']:
        if sem_type == 'III_V':
            resultado = MATERIAL_PERMUTATOR.generate_iii_v_combinations(filtros_led)
        else:
            resultado = MATERIAL_PERMUTATOR.generate_ii_vi_combinations(filtros_led)
        
        # Filtrar por band gap para LED azul
        for semiconductor in resultado.filtered_combinations:
            if semiconductor.properties and semiconductor.properties.band_gap:
                bg = semiconductor.properties.band_gap
                if 2.5 <= bg <= 3.5:
                    candidatos.append(semiconductor)
    
    print(f"🎯 Candidatos para LED azul: {len(candidatos)}")
    
    # Mostrar top candidatos
    candidatos.sort(key=lambda x: x.properties.band_gap if x.properties else 0)
    
    print("\n📋 Top candidatos:")
    for i, candidato in enumerate(candidatos[:10]):
        bg = candidato.properties.band_gap if candidato.properties else 'N/A'
        lattice = candidato.estimate_lattice_constant()
        print(f"   {i+1:2d}. {candidato.formula:8s} - "
              f"BG: {bg:4.2f} eV, "
              f"a: {lattice:.3f} Å")
    
    return candidatos

candidatos_led = screening_led_azul()
```

### 📊 Caso de Uso 3: Análisis de Datos

**Escenario**: Analizar datos existentes y generar reportes ejecutivos.

```python
# caso_analisis.py
from analysis.multi_material_analysis import MultiMaterialAnalyzer
from pathlib import Path

def analizar_resultados_existentes():
    print("📊 Análisis de Resultados Existentes")
    print("=" * 38)
    
    # Buscar resultados anteriores
    resultados_dir = Path("results")
    
    if not resultados_dir.exists():
        print("❌ No se encontraron resultados previos")
        return None
    
    # Cargar datos de campañas anteriores
    campaign_files = list(resultados_dir.glob("campaign_*/campaign_summary.json"))
    
    if not campaign_files:
        print("❌ No se encontraron datos de campañas")
        return None
    
    print(f"📁 Encontradas {len(campaign_files)} campañas")
    
    # Analizar cada campaña
    analyzer = MultiMaterialAnalyzer()
    all_results = []
    
    for campaign_file in campaign_files:
        print(f"📖 Analizando: {campaign_file.parent.name}")
        
        # Simular carga de resultados (en implementación real, cargar desde JSON)
        # Por ahora, crear datos de ejemplo
        from workflow.multi_material_pipeline import CampaignResult
        
        # ... código para cargar datos reales ...
        
        print(f"   ✅ Análisis completado")
    
    return all_results

resultados = analizar_resultados_existentes()
```

---

## 5. Mejores Prácticas y Recomendaciones

### ✅ Mejores Prácticas Generales

#### 1. Planificación de Recursos
```python
# ✅ CORRECTO: Calcular workers según recursos
import multiprocessing
import psutil

cpu_cores = multiprocessing.cpu_count()
available_memory_gb = psutil.virtual_memory().available / 1024**3
optimal_workers = min(cpu_cores, int(available_memory_gb / 4))

print(f"Workers óptimos: {optimal_workers}")
```

#### 2. Configuración de Filtros
```python
# ✅ CORRECTO: Filtros conservadores para materiales desconocidos
filtros_seguros = PermutationFilter(
    max_ionic_radius_ratio=2.0,        # Más restrictivo
    min_electronegativity_diff=0.5,    # Más restrictivo
    only_common_elements=True,          # Solo elementos conocidos
    exclude_toxic=True,                 # Excluir tóxicos
    exclude_radioactive=True            # Excluir radiactivos
)
```

#### 3. Gestión de Resultados
```python
# ✅ CORRECTO: Organización clara de resultados
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(f"results/estudio_{timestamp}")
results_dir.mkdir(parents=True, exist_ok=True)

# Guardar todo en directorio organizado
```

### ⚠️ Errores Comunes a Evitar

#### 1. Sobre-paralelización
```python
# ❌ INCORRECTO: Demasiados workers
pipeline.set_parallel_workers(32)  # Para sistema de 8 cores

# ✅ CORRECTO: Workers según recursos
workers = min(cpu_cores * 2, 8)  # Máximo 8 para estabilidad
pipeline.set_parallel_workers(workers)
```

#### 2. Filtros Demasiado Laxos
```python
# ❌ INCORRECTO: Filtros muy permisivos
filtros_permisivos = PermutationFilter(
    max_ionic_radius_ratio=10.0,  # Demasiado alto
    exclude_toxic=False,           # Incluir tóxicos
    exclude_radioactive=False      # Incluir radiactivos
)

# ✅ CORRECTO: Filtros balanceados
filtros_balanceados = PermutationFilter(
    max_ionic_radius_ratio=2.5,
    exclude_toxic=True,
    exclude_radioactive=True
)
```

#### 3. No Validar Datos
```python
# ❌ INCORRECTO: Asumir que los datos son válidos
material = SEMICONDUCTOR_DB.get_semiconductor('GaAs')
lattice = material.properties.lattice_constant  # Sin verificar

# ✅ CORRECTO: Validar datos antes de usar
material = SEMICONDUCTOR_DB.get_semiconductor('GaAs')
if material and material.properties and material.properties.lattice_constant:
    lattice = material.properties.lattice_constant
else:
    print("⚠️  Usando valor por defecto")
    lattice = 5.65  # Valor de referencia
```

### 📈 Optimización de Rendimiento

#### 1. Para Estudios Grandes (>20 materiales)
```python
# Configuración optimizada para estudios masivos
config_produccion = MultiMaterialConfig(
    parallel_materials=True,
    max_concurrent_materials=8,
    memory_limit_gb=32.0,
    auto_cleanup=True,
    checkpoint_interval_minutes=30
)
```

#### 2. Para Desarrollo y Pruebas
```python
# Configuración rápida para desarrollo
config_desarrollo = MultiMaterialConfig(
    parallel_materials=False,  # Secuencial para debugging
    max_concurrent_materials=1,
    auto_cleanup=False,        # Mantener resultados
    verbose_logging=True
)
```

#### 3. Gestión de Memoria
```python
# Monitoreo automático de memoria
pipeline = MultiMaterialPipeline(config)
pipeline.memory_monitoring_enabled = True
pipeline.memory_limit_gb = psutil.virtual_memory().available * 0.8 / 1024**3
pipeline.memory_reduction_factor = 0.5  # Reducir workers si es necesario
```

---

## 6. Troubleshooting Común

### 🔧 Problemas de Instalación

#### Error: "ModuleNotFoundError: No module named 'src'"
```bash
# Solución: Agregar src al PYTHONPATH
export PYTHONPATH="/path/to/preconvergencia-gaAs:$PYTHONPATH"

# O en código Python:
import sys
sys.path.insert(0, "/path/to/preconvergencia-gaAs")
```

#### Error: "Permission denied" durante instalación
```bash
# Solución: Usar --user para instalación local
pip install --user -r requirements.txt

# O crear entorno virtual
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Error: "Docker: command not found"
```bash
# Solución: Instalar Docker
# Ubuntu/Debian:
sudo apt update
sudo apt install docker.io

# macOS: Descargar Docker Desktop
# Windows: Usar WSL2 con Docker Desktop
```

### 🚨 Problemas de Ejecución

#### Error: "Out of Memory" durante ejecución paralela
```python
# Diagnóstico
import psutil
print(f"RAM disponible: {psutil.virtual_memory().available / 1024**3:.1f} GB")

# Solución 1: Reducir workers
pipeline.set_parallel_workers(2)

# Solución 2: Usar modo secuencial
pipeline.enable_parallel_execution(False)

# Solución 3: Reducir memoria por material
pipeline.memory_limit_gb = 4.0  # 4GB máximo
```

#### Error: "Material not found in database"
```python
# Diagnóstico
from models.semiconductor_database import SEMICONDUCTOR_DB

# Verificar si el material existe
material = 'GaAs'
if material in SEMICONDUCTOR_DB.semiconductors:
    print("Material existe")
else:
    print("Material no existe")

# Solución 1: Usar nombre correcto
materiales_validos = list(SEMICONDUCTOR_DB.semiconductors.keys())
print("Materiales disponibles:", materiales_validos[:10])

# Solución 2: Generar material automáticamente
from core.material_permutator import generate_all_iii_v
result = generate_all_iii_v()
# El material se agregará automáticamente si pasa los filtros
```

#### Error: "Convergence failed" o "Calculation diverged"
```python
# Diagnóstico: Verificar configuración
print(f"Cutoff list: {config.cutoff_list}")
print(f"K-mesh list: {config.kmesh_list}")
print(f"Lattice constant: {config.lattice_constant}")

# Solución 1: Ajustar parámetros iniciales
config.cutoff_list = [300, 400, 500]  # Valores más conservadores
config.kmesh_list = [(2,2,2), (4,4,4)]  # Mallas menos densas
config.lattice_constant = 5.65  # Valor inicial más cercano

# Solución 2: Aumentar tolerancias
config.energy_convergence = 1e-4  # Menos estricta
config.max_scf_iterations = 100    # Más iteraciones

# Solución 3: Usar parámetros específicos del material
from models.semiconductor_database import SEMICONDUCTOR_DB
material_data = SEMICONDUCTOR_DB.get_semiconductor('GaAs')
if material_data and material_data.properties.lattice_constant:
    config.lattice_constant = material_data.properties.lattice_constant
```

### 🐛 Problemas de Rendimiento

#### Ejecución muy lenta
```python
# Diagnóstico: Verificar configuración
print(f"Workers: {pipeline.config.max_concurrent_materials}")
print(f"Paralelización: {pipeline.config.parallel_materials}")

# Solución 1: Habilitar paralelización
pipeline.enable_parallel_execution(True)

# Solución 2: Ajustar número de workers
pipeline.set_parallel_workers(min(8, multiprocessing.cpu_count()))

# Solución 3: Usar configuración más rápida
from config.settings import get_fast_config
config = get_fast_config()  # Menos puntos de cálculo
```

#### Muchos archivos temporales
```python
# Solución: Limpiar archivos temporales
import shutil
import tempfile

# Limpiar directorio temporal
temp_dir = Path(tempfile.gettempdir()) / "preconvergencia"
if temp_dir.exists():
    shutil.rmtree(temp_dir)
    print("Archivos temporales limpiados")

# Configurar limpieza automática
pipeline.auto_cleanup = True
```

### 📊 Problemas de Análisis

#### Resultados inconsistentes
```python
# Diagnóstico: Verificar semilla aleatoria y configuración
import random
random.seed(42)  # Establecer semilla

# Verificar configuración
print("Configuración de cálculo:")
for key, value in config.__dict__.items():
    print(f"  {key}: {value}")

# Solución: Usar configuración reproducible
config.reproducible = True
config.random_seed = 42
```

#### Análisis estadístico falla
```python
# Diagnóstico: Verificar datos de entrada
from analysis.multi_material_analysis import MultiMaterialAnalyzer

analyzer = MultiMaterialAnalyzer()
validation = analyzer.validate_input_data(campaign_result)

if not validation.is_valid:
    print(f"Errores de validación: {validation.errors}")
    print(f"Advertencias: {validation.warnings}")

# Solución: Filtrar datos válidos
valid_results = [r for r in campaign_result.individual_results if r.success]
print(f"Resultados válidos: {len(valid_results)}")
```

---

## 7. Preguntas Frecuentes (FAQ)

### ❓ Preguntas Generales

**P: ¿Cuál es la diferencia entre el sistema original y el multimaterial?**
R: El sistema original estaba diseñado para un solo material (GaAs), mientras que el sistema multimaterial puede procesar múltiples semiconductores simultáneamente, generar combinaciones automáticamente y realizar análisis comparativos.

**P: ¿Puedo usar mis configuraciones existentes del sistema original?**
R: Sí, existe un workflow de migración automática. Ver `docs/WORKFLOWS_OPTIMIZADOS.md` sección 6.

**P: ¿Qué tipos de semiconductores soporta?**
R: Actualmente soporta semiconductores III-V y II-VI. El roadmap incluye IV-IV, ternarios y cuaternarios.

### 🔬 Preguntas Técnicas

**P: ¿Cuántos materiales puedo procesar simultáneamente?**
R: Depende de tus recursos. Como regla: 1-2 workers por CPU core, con 4GB RAM por worker activo.

**P: ¿Los resultados son reproducibles?**
R: Sí, con `config.reproducible = True` y `config.random_seed = valor`.

**P: ¿Puedo integrar con mi código DFT existente?**
R: Sí, reemplaza `src/core/calculator.py` manteniendo la interfaz `calculate_energy()`.

**P: ¿Soporta otros códigos DFT además de PySCF?**
R: La arquitectura está diseñada para ser independiente del código DFT. Solo necesitas implementar la interfaz de cálculo.

### 💾 Preguntas de Datos

**P: ¿De dónde vienen los datos experimentales?**
R: De literatura científica peer-reviewed, Materials Project y bases de datos experimentales validadas.

**P: ¿Puedo agregar mis propios datos?**
R: Sí, mediante la API `SEMICONDUCTOR_DB.add_semiconductor()` o cargando CSV personalizados.

**P: ¿Cómo se valida la calidad de los datos?**
R: Sistema de validación automática que verifica consistencia química y física.

### 🚀 Preguntas de Rendimiento

**P: ¿Cuál es el rendimiento típico?**
R: Material individual: 5-15 min, 5 materiales en paralelo: 15-30 min, 10 materiales: 30-60 min.

**P: ¿Funciona en supercomputadoras?**
R: Sí, configuración optimizada en `config/hpc.yaml` para SLURM, PBS, etc.

**P: ¿Necesito GPU para mejor rendimiento?**
R: No necesario para preconvergencia. GPU sería útil para cálculos DFT posteriores.

### 🛠️ Preguntas de Desarrollo

**P: ¿Cómo agrego nuevos tipos de semiconductores?**
R: Extiende `SemiconductorType` enum, agrega elementos a `periodic_table_groups.py`, implementa generador en `material_permutator.py`.

**P: ¿Puedo personalizar los filtros de generación?**
R: Sí, crea filtros personalizados con `PermutationFilter(custom_filters=[mi_filtro])`.

**P: ¿Cómo integro con otras herramientas?**
R: Sistema modular con APIs claras. Ver ejemplos en `examples/` para integración.

### 📋 Preguntas de Configuración

**P: ¿Qué archivo de configuración debo usar?**
R: `config/default.yaml` para desarrollo, `config/production.yaml` para producción, `config/hpc.yaml` para clusters.

**P: ¿Puedo tener configuraciones específicas por material?**
R: Sí, `MultiMaterialConfig` permite parámetros específicos por semiconductor.

**P: ¿Cómo configuro logging personalizado?**
R: Crea `config/logging.yaml` o configura via código con `setup_logging()`.

### 🎯 Preguntas de Casos de Uso

**P: ¿Es adecuado para estudios de alta escala?**
R: Sí, optimizado para estudios de 50+ materiales con paralelización masiva.

**P: ¿Genera reportes para publicaciones?**
R: Sí, incluye análisis estadístico, visualizaciones y formatos exportables para papers.

**P: ¿Puedo usarlo para diseño de heteroestructuras?**
R: Sí, sistema integrado de búsqueda de matching de constantes de red.

### ❓ Preguntas de Soporte

**P: ¿Dónde reporto bugs?**
R: GitHub Issues del proyecto, con logs y configuración de ejemplo.

**P: ¿Hay comunidad activa?**
R: Sí, GitHub Discussions para preguntas, Discord para chat en tiempo real.

**P: ¿Hay training oficial?**
R: Tutoriales en video en progreso, workshops en conferencias científicas.

### 🔮 Preguntas de Futuro

**P: ¿Qué hay en el roadmap?**
R: Soporte para ternarios/cuaternarios (Q1 2025), ML para predicción de propiedades (Q2 2025), interfaz web (Q3 2025).

**P: ¿Será open source?**
R: Sí, licencia MIT. Código disponible en GitHub.

**P: ¿Soporte comercial disponible?**
R: En evaluación. Contactar para detalles de enterprise support.

---

## 📞 Obtener Ayuda Adicional

### Canales de Soporte
- **Documentación**: Este archivo y `docs/`
- **Ejemplos**: Carpeta `examples/`
- **GitHub Issues**: Para bugs y features
- **GitHub Discussions**: Para preguntas
- **Email**: support@preconvergencia.org

### Contribuir
- **Reportar bugs**: GitHub Issues
- **Solicitar features**: GitHub Discussions
- **Contribuir código**: Pull Requests
- **Mejorar documentación**: Issues con label "docs"

### Mantenerse Actualizado
- **Releases**: GitHub Releases
- **Changelog**: CHANGELOG.md
- **Blog**: blog.preconvergencia.org
- **Twitter**: @Preconvergencia

---

## 🎉 ¡Feliz Investigación!

Esperamos que esta guía te ayude a maximizar el potencial del sistema de preconvergencia multimaterial. ¡Que disfrutes descubriendo nuevos materiales semiconductores!

**¿Listo para tu primer análisis?** Comienza con el [Tutorial 1](#-tutorial-1-tu-primer-análisis-15-minutos) y explora desde ahí.

---

*Última actualización: Noviembre 2024*  
*Versión: 2.0*  
*Documentación del Sistema de Preconvergencia Multimaterial*