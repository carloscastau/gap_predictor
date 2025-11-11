# Plan de Acción Detallado para Preconvergencia-GaAs v2.0

## 📋 Información del Proyecto
- **Proyecto**: Pipeline modular de preconvergencia DFT/PBC para GaAs
- **Estado Actual**: 85% funcional, problemas de imports y dependencias
- **Tiempo Estimado de Implementación**: 1-2 semanas
- **Prioridad**: Alta - Proyecto crítico para investigación

---

## 🚀 ACCIÓN INMEDIATA (0-1 hora)

### 1. Instalación Completa de PySCF y Dependencias de Compilación

#### 1.1 Diagnóstico del Sistema Actual
```bash
# Verificar herramientas de compilación disponibles
which gfortran gcc cmake make
gfortran --version
gcc --version
cmake --version

# Verificar entorno Python actual
./venv/bin/python --version
./venv/bin/pip list | grep -E "(numpy|scipy|matplotlib|pymatgen|spglib)"
```

#### 1.2 Instalación de Dependencias del Sistema
```bash
# UBUNTU/DEBIAN
sudo apt-get update
sudo apt-get install -y \
    gfortran \
    cmake \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    pkg-config

# CENTOS/RHEL
sudo yum install -y \
    gcc-gfortran \
    cmake \
    gcc \
    make \
    openblas-devel \
    openmpi-devel

# Verificar instalación
gfortran --version
cmake --version
```

#### 1.3 Instalación Completa de PySCF
```bash
# Activar entorno virtual
source venv/bin/activate

# Desinstalar versión previa si existe
pip uninstall pyscf -y

# Instalar PySCF completo
pip install pyscf==2.3.0 --no-binary=pyscf

# Verificar instalación
python -c "
from pyscf.pbc import gto, dft
print('✅ PySCF instalado correctamente')
print(f'Versión PySCF: {__import__(\"pyscf\").__version__}')
"

# Verificar funcionalidad básica
python -c "
import numpy as np
from pyscf.pbc import gto, dft

# Test básico de construcción de celda
cell = gto.Cell()
cell.atom = 'C 0 0 0'
cell.basis = 'gth-dzvp'
cell.a = np.eye(3) * 3.5668
cell.build()
print('✅ PySCF funcional - construcción de celda exitosa')
"
```

### 2. Diagnóstico Completo de Imports y Estructura

#### 2.1 Script de Diagnóstico de Imports
```python
#!/usr/bin/env python3
"""
diagnostico_imports.py - Diagnóstico completo de la estructura de imports
"""

import sys
import importlib
import traceback
from pathlib import Path
import ast

def analizar_estructura_modulos():
    """Analiza la estructura de módulos del proyecto."""
    src_path = Path("src")
    problemas = []
    
    for py_file in src_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                contenido = f.read()
            
            # Parsear el archivo
            tree = ast.parse(contenido)
            
            # Buscar imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        problema = verificar_import(alias.name, py_file)
                        if problema:
                            problemas.append(problema)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        nivel = node.level if node.level else 0
                        modulo = node.module
                        problema = verificar_import_from(modulo, nivel, py_file)
                        if problema:
                            problemas.append(problema)
                            
        except Exception as e:
            problemas.append(f"ERROR: {py_file}: {e}")
    
    return problemas

def verificar_import(modulo, archivo_origen):
    """Verifica un import simple."""
    if modulo.startswith('.'):
        return f"Import relativo en {archivo_origen}: {modulo}"
    return None

def verificar_import_from(modulo, nivel, archivo_origen):
    """Verifica un import desde módulo."""
    if nivel > 0:
        return f"Import relativo {nivel} niveles en {archivo_origen}: {modulo or '(módulo vacío)'}"
    return None

def main():
    print("🔍 Iniciando diagnóstico de imports...")
    problemas = analizar_estructura_modulos()
    
    if problemas:
        print("❌ Problemas encontrados:")
        for i, problema in enumerate(problemas, 1):
            print(f"{i:3d}. {problema}")
    else:
        print("✅ No se encontraron problemas de imports")
    
    return len(problemas)

if __name__ == "__main__":
    exit(main())
```

#### 2.2 Ejecución del Diagnóstico
```bash
# Ejecutar diagnóstico
./venv/bin/python diagnostico_imports.py

# Verificar estructura de archivos
find src -name "*.py" | head -20
ls -la src/workflow/checkpoint/
```

### 3. Verificación de Integridad de Archivos
```bash
# Verificar archivos críticos
ls -la src/workflow/checkpoint/checkpoint.py
ls -la src/workflow/pipeline.py
ls -la src/core/calculator.py
ls -la src/config/settings.py

# Verificar que todos los __init__.py existen
find src -name "__init__.py" -exec echo "✅ {}" \;
```

---

## 📅 CORTO PLAZO (1-3 días)

### 1. Refactorización Sistemática de Imports Circulares

#### 1.1 Análisis de Dependencias Circulares
```python
#!/usr/bin/env python3
"""
analizador_circular_imports.py - Identifica y resuelve imports circulares
"""

import sys
import ast
from pathlib import Path
from collections import defaultdict, deque

class AnalizadorImports:
    def __init__(self):
        self.deps = defaultdict(set)
        self.archivos = {}
    
    def analizar_archivo(self, archivo):
        """Analiza imports en un archivo."""
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
            
            tree = ast.parse(contenido)
            modulo_nombre = self.obtener_nombre_modulo(archivo)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level > 0:  # Import relativo
                        modulo_origen = self.resolver_import_relativo(
                            modulo_nombre, node.module, node.level
                        )
                        if modulo_origen:
                            self.deps[modulo_nombre].add(modulo_origen)
                            
        except Exception as e:
            print(f"Error analizando {archivo}: {e}")
    
    def encontrar_circulares(self):
        """Encuentra dependencias circulares."""
        circulares = []
        visitados = set()
        en_pila = set()
        
        def dfs(nodo, camino):
            if nodo in en_pila:
                # Encontramos un ciclo
                idx = camino.index(nodo)
                ciclo = camino[idx:] + [nodo]
                circulares.append(ciclo)
                return
            
            if nodo in visitados:
                return
            
            visitados.add(nodo)
            en_pila.add(nodo)
            
            for dependencia in self.deps.get(nodo, []):
                if dependencia in self.archivos:
                    dfs(dependencia, camino + [nodo])
            
            en_pila.remove(nodo)
        
        for modulo in self.archivos:
            if modulo not in visitados:
                dfs(modulo, [])
        
        return circulares
    
    def obtener_nombre_modulo(self, archivo):
        """Convierte path de archivo a nombre de módulo."""
        rel_path = Path(archivo).relative_to(Path("src"))
        return str(rel_path.with_suffix('')).replace('/', '.')
    
    def resolver_import_relativo(self, modulo_actual, modulo_importado, nivel):
        """Resuelve import relativo."""
        if not modulo_importado:
            # Import del mismo módulo
            partes = modulo_actual.split('.')
            if nivel <= len(partes):
                return '.'.join(partes[:-nivel]) if nivel > 0 else modulo_actual
        return None

def main():
    analizador = AnalizadorImports()
    
    # Analizar todos los archivos Python
    for py_file in Path("src").rglob("*.py"):
        if "__pycache__" not in str(py_file):
            analizador.archivos[analizador.obtener_nombre_modulo(py_file)] = py_file
            analizador.analizar_archivo(py_file)
    
    # Encontrar circulares
    circulares = analizador.encontrar_circulares()
    
    if circulares:
        print("❌ Dependencias circulares encontradas:")
        for i, ciclo in enumerate(circulares, 1):
            print(f"{i}. {' → '.join(ciclo)}")
    else:
        print("✅ No se encontraron dependencias circulares")
    
    return len(circulares)

if __name__ == "__main__":
    exit(main())
```

#### 1.2 Estrategia de Resolución de Circulares

**Patrón 1: Late Binding**
```python
# En lugar de:
from .module import function

# Usar:
def get_function():
    from .module import function
    return function
```

**Patrón 2: Interface Segregation**
```python
# Crear interfaces separadas
# src/interfaces/__init__.py
from .calculator import ICalculator
from .optimizer import IOptimizer

# src/core/calculator.py
from ..interfaces import ICalculator

class DFTCalculator(ICalculator):
    # Implementación
```

**Patrón 3: Dependency Injection**
```python
# En lugar de imports directos
class Pipeline:
    def __init__(self, calculator_class=None, optimizer_class=None):
        self.calculator = calculator_class or DFTCalculator
        self.optimizer = optimizer_class or LatticeOptimizer
```

#### 1.3 Implementación de Correcciones
```bash
# Crear backup antes de refactorizar
cp -r src src_backup_$(date +%Y%m%d_%H%M%S)

# Aplicar correcciones sistemáticamente
# 1. Resolver imports en checkpoint
# 2. Resolver imports en pipeline  
# 3. Resolver imports en calculator
# 4. Verificar con tests
```

### 2. Verificación de Ubicación de Archivos de Módulo

#### 2.1 Auditoría de Estructura
```bash
#!/bin/bash
# auditoria_estructura.sh

echo "🔍 Auditoría de estructura de módulos..."

# Verificar estructura esperada
ESTRUCTURA_ESPERADA=(
    "src/config/settings.py"
    "src/config/validation.py"
    "src/config/hardware.py"
    "src/core/calculator.py"
    "src/core/optimizer.py"
    "src/core/parallel.py"
    "src/models/basis.py"
    "src/models/cell.py"
    "src/models/kpoints.py"
    "src/models/results.py"
    "src/workflow/pipeline.py"
    "src/workflow/checkpoint/__init__.py"
    "src/workflow/checkpoint/checkpoint.py"
    "src/workflow/stages/__init__.py"
    "src/workflow/stages/base.py"
    "src/analysis/statistics.py"
    "src/visualization/plots.py"
    "src/utils/logging.py"
)

echo "Verificando archivos críticos..."
for archivo in "${ESTRUCTURA_ESPERADA[@]}"; do
    if [ -f "src/$archivo" ]; then
        echo "✅ $archivo"
    else
        echo "❌ FALTA: $archivo"
    fi
done

# Verificar __init__.py
echo -e "\nVerificando __init__.py..."
find src -type d -exec test -f {}/__init__.py \; -print | while read dir; do
    echo "✅ $dir/__init__.py"
done

find src -type d ! -exec test -f {}/__init__.py \; -print | while read dir; do
    echo "❌ FALTA: $dir/__init__.py"
done
```

#### 2.2 Corrección de Ubicaciones
```bash
# Si faltan archivos, crearlos o moverlos
# Ejemplo: verificar que checkpoint.py esté en lugar correcto
if [ ! -f "src/workflow/checkpoint/checkpoint.py" ] && [ -f "src/workflow/checkpoint.py" ]; then
    echo "Moviendo checkpoint.py a ubicación correcta..."
    mv src/workflow/checkpoint.py src/workflow/checkpoint/checkpoint.py
fi
```

### 3. Suite de Tests Automatizada

#### 3.1 Tests de Integridad de Imports
```python
# tests/test_imports_integrity.py
import pytest
import importlib
import sys
from pathlib import Path

class TestImportsIntegrity:
    """Tests para verificar integridad de imports."""
    
    @pytest.mark.parametrize("module_name", [
        "config.settings",
        "core.calculator", 
        "core.optimizer",
        "core.parallel",
        "workflow.pipeline",
        "workflow.checkpoint",
        "utils.logging",
        "analysis.statistics",
        "visualization.plots"
    ])
    def test_module_imports(self, module_name):
        """Test que cada módulo se puede importar sin errores."""
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_no_circular_imports(self):
        """Test que no hay imports circulares evidentes."""
        # Lista de módulos principales
        modulos = [
            "config.settings",
            "core.calculator",
            "core.optimizer", 
            "core.parallel",
            "workflow.pipeline",
            "workflow.checkpoint",
            "utils.logging"
        ]
        
        # Intentar importar todos en secuencia
        for modulo in modulos:
            try:
                importlib.import_module(modulo)
            except ImportError as e:
                pytest.fail(f"Import error in {modulo}: {e}")
    
    def test_config_classes_available(self):
        """Test que las clases de configuración están disponibles."""
        from config.settings import PreconvergenceConfig
        
        # Test creación básica
        config = PreconvergenceConfig()
        assert config.lattice_constant > 0
        assert len(config.cutoff_list) > 0
    
    def test_core_classes_available(self):
        """Test que las clases core están disponibles."""
        from core.calculator import DFTCalculator, CellParameters
        from core.optimizer import LatticeOptimizer, ConvergenceAnalyzer
        from core.parallel import MemoryMonitor, TaskScheduler
        
        # Test instanciación básica
        from config.settings import PreconvergenceConfig
        config = PreconvergenceConfig()
        
        calc = DFTCalculator(config)
        opt = LatticeOptimizer(config)
        monitor = MemoryMonitor()
        scheduler = TaskScheduler(config)
        
        assert calc is not None
        assert opt is not None
        assert monitor is not None
        assert scheduler is not None
```

#### 3.2 Tests de Funcionalidad Básica
```python
# tests/test_functionality.py
import pytest
import asyncio
from pathlib import Path

class TestBasicFunctionality:
    """Tests de funcionalidad básica."""
    
    @pytest.mark.asyncio
    async def test_calculator_creation(self):
        """Test creación de calculadora."""
        from config.settings import PreconvergenceConfig
        from core.calculator import DFTCalculator, CellParameters
        
        config = PreconvergenceConfig()
        calc = DFTCalculator(config)
        
        # Crear parámetros de celda
        cell_params = CellParameters(
            lattice_constant=5.653,
            x_ga=0.25,
            cutoff=80.0,
            kmesh=(2, 2, 2),
            basis="gth-dzvp",
            pseudo="gth-pbe",
            xc="PBE",
            sigma_ha=0.01,
            conv_tol=1e-8
        )
        
        assert cell_params.lattice_constant == 5.653
        assert cell_params.estimated_memory > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_creation(self):
        """Test creación de pipeline."""
        from config.settings import PreconvergenceConfig
        from workflow.pipeline import PreconvergencePipeline
        
        config = PreconvergenceConfig()
        pipeline = PreconvergencePipeline(config)
        
        assert pipeline is not None
        assert len(pipeline.stages) > 0
    
    def test_configuration_validation(self):
        """Test validación de configuración."""
        from config.settings import PreconvergenceConfig
        
        # Configuración válida
        config = PreconvergenceConfig(
            lattice_constant=5.653,
            cutoff_list=[80, 120, 160]
        )
        assert config is not None
        
        # Configuración inválida debe lanzar excepción
        with pytest.raises(ValueError):
            PreconvergenceConfig(lattice_constant=10.0)  # Fuera de rango
    
    def test_checkpoint_system(self):
        """Test sistema de checkpoints."""
        from workflow.checkpoint import CheckpointManager
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Test guardar checkpoint
            data = {"test": "data", "number": 42}
            filepath = manager.save_checkpoint("test_stage", data)
            
            assert filepath.exists()
            
            # Test cargar checkpoint
            loaded_data = manager.load_checkpoint("test_stage")
            assert loaded_data is not None
            assert loaded_data["data"]["test"] == "data"
```

#### 3.3 Ejecución de Verificación Final
```bash
# Ejecutar suite completa
./venv/bin/python -m pytest tests/ -v --tb=short

# Con coverage
./venv/bin/python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Solo tests de imports
./venv/bin/python -m pytest tests/test_imports_integrity.py -v

# Solo tests de funcionalidad
./venv/bin/python -m pytest tests/test_functionality.py -v

# Tests específicos por módulo
./venv/bin/python -m pytest tests/test_imports_integrity.py::TestImportsIntegrity::test_module_imports -v
```

---

## 📊 MEDIO PLAZO (1-2 semanas)

### 1. Optimizaciones de Rendimiento

#### 1.1 Optimización de Imports

**A. Implementar Lazy Loading**
```python
# src/core/calculator.py
class DFTCalculator:
    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self._pyscf_loaded = False
        self._pymatgen_loaded = False
    
    def _load_pyscf(self):
        """Lazy loading de PySCF."""
        if not self._pyscf_loaded:
            from pyscf import lib as pyscf_lib
            from pyscf.pbc import gto, dft
            from pyscf.scf import addons as scf_addons
            self._pyscf_lib = pyscf_lib
            self._pyscf_gto = gto
            self._pyscf_dft = dft
            self._pyscf_addons = scf_addons
            self._pyscf_loaded = True
    
    def _load_pymatgen(self):
        """Lazy loading de PyMatGen."""
        if not self._pymatgen_loaded:
            from pymatgen.core.surface import HighSymmKpath
            self._pymatgen_kpath = HighSymmKpath
            self._pymatgen_loaded = True
```

**B. Caching de Módulos Importados**
```python
# src/utils/module_cache.py
import functools
import importlib
from typing import Dict, Any

class ModuleCache:
    """Cache de módulos importados para mejorar rendimiento."""
    
    _instance = None
    _cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_module(self, module_name: str):
        """Obtiene módulo con cache."""
        if module_name not in self._cache:
            try:
                self._cache[module_name] = importlib.import_module(module_name)
            except ImportError:
                self._cache[module_name] = None
        return self._cache[module_name]

# Decorador para caching de imports
def cached_import(module_path: str):
    """Decorador para imports con cache."""
    def decorator(func):
        cache = ModuleCache()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module = cache.get_module(module_path)
            if module is None:
                raise ImportError(f"Cannot import {module_path}")
            return func(module, *args, **kwargs)
        
        return wrapper
    return decorator

# Uso en código
@cached_import("pyscf.pbc.dft")
def create_dft_calculator(module, cell, kpts):
    return module.KRKS(cell, kpts=kpts)
```

**C. Optimización de Inicialización**
```python
# src/core/calculator.py
class DFTCalculator:
    __slots__ = ['config', '_initialized', '_pyscf_modules']
    
    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self._initialized = False
        self._pyscf_modules = {}
    
    def _initialize_pyscf(self):
        """Inicialización lazy de PySCF con optimización."""
        if self._initialized:
            return
        
        # Cargar solo módulos necesarios
        try:
            from pyscf import lib as pyscf_lib
            
            # Configurar límites de memoria
            pyscf_lib.param.MAX_MEMORY = self.config.memory_limit_gb * 1024**3
            
            # Configurar paralelismo
            pyscf_lib.num_threads(1)
            
            # Configurar tolerancias
            pyscf_lib.param.TOLERANCE = 1e-10
            
            # Cachear módulos
            self._pyscf_modules = {
                'lib': pyscf_lib,
                'gto': pyscf_lib,
                'dft': pyscf_lib,
                'addons': pyscf_lib
            }
            
            self._initialized = True
            
        except ImportError:
            # Fallback para simulación
            self._pyscf_modules = {}
            self._initialized = True
```

#### 1.2 Optimización de Carga de Módulos

**A. Preloading Strategy**
```python
# src/core/preload.py
"""Estrategia de precarga de módulos."""

import importlib
import asyncio
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor

class ModulePreloader:
    """Precarga módulos en background para mejorar performance."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preloaded_modules: Set[str] = set()
    
    def preload_core_modules(self):
        """Preload módulos core del sistema."""
        core_modules = [
            'config.settings',
            'core.calculator',
            'core.optimizer', 
            'core.parallel',
            'utils.logging'
        ]
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.preloaded_modules.add(module_name)
            except ImportError as e:
                print(f"Warning: Could not preload {module_name}: {e}")
    
    async def preload_async(self, modules: List[str]):
        """Preload módulos de forma asíncrona."""
        tasks = []
        for module_name in modules:
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, 
                importlib.import_module, 
                module_name
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_preloaded_modules(self) -> List[str]:
        """Retorna lista de módulos precargados."""
        return list(self.preloaded_modules)

# Inicialización global
_preloader = ModulePreloader()
_preloader.preload_core_modules()
```

**B. Import Optimization Patterns**
```python
# src/utils/import_helpers.py
"""Helpers para optimización de imports."""

import functools
from typing import Optional, Callable

class LazyImport:
    """Wrapper para imports lazy optimizado."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module: Optional[object] = None
    
    def __getattr__(self, name: str):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        return self._module(*args, **kwargs)

# Patrón de uso optimizado
pyscf = LazyImport("pyscf")
pymatgen = LazyImport("pymatgen")

# En lugar de:
# from pyscf.pbc import dft
# from pymatgen.core.surface import HighSymmKpath

# Usar:
# kmf = pyscf.pbc.dft.KRKS(cell, kpts=kpts)
# kpath = pymatgen.core.surface.HighSymmKpath(structure)
```

### 2. Sistema de Monitoreo Continuo

#### 2.1 Performance Monitor
```python
# src/utils/performance_monitor.py
import time
import psutil
import threading
from typing import Dict, List, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import json

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    import_time_ms: float
    module_count: int

class PerformanceMonitor:
    """Monitor continuo de rendimiento."""
    
    def __init__(self, window_size: int = 100):
        self.metrics: deque = deque(maxlen=window_size)
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable] = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Inicia monitoreo en background."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Detiene monitoreo."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Loop principal de monitoreo."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Notificar callbacks
                for callback in self.callbacks:
                    callback(metrics)
                
                time.sleep(interval)
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Recolecta métricas actuales."""
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_mb=psutil.virtual_memory().used / (1024 * 1024),
            import_time_ms=self._measure_import_time(),
            module_count=len(sys.modules)
        )
    
    def _measure_import_time(self) -> float:
        """Mide tiempo de imports recientes."""
        # Implementación simplificada
        return 0.0  # Placeholder
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Retorna métricas actuales."""
        if self.metrics:
            return self.metrics[-1]
        return None
    
    def get_performance_summary(self) -> Dict:
        """Retorna resumen de rendimiento."""
        if not self.metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_mb for m in self.metrics]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'max_memory_mb': max(memory_values),
            'samples_collected': len(self.metrics),
            'monitoring_duration': self.metrics[-1].timestamp - self.metrics[0].timestamp
        }
    
    def save_metrics(self, filepath: str):
        """Guarda métricas a archivo."""
        data = {
            'metrics': [vars(m) for m in self.metrics],
            'summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
```

#### 2.2 Benchmark Suite
```python
# tests/benchmarks/performance_benchmarks.py
import pytest
import time
import psutil
import memory_profiler
from pathlib import Path
from src.utils.performance_monitor import PerformanceMonitor

class TestPerformanceBenchmarks:
    """Benchmarks de rendimiento del sistema."""
    
    @pytest.mark.benchmark
    def test_module_import_benchmark(self, benchmark):
        """Benchmark de velocidad de imports."""
        def import_all_modules():
            modules = [
                'config.settings',
                'core.calculator',
                'core.optimizer',
                'core.parallel',
                'workflow.pipeline',
                'utils.logging'
            ]
            for module in modules:
                try:
                    __import__(module)
                except ImportError:
                    pass  # Skip missing modules in benchmark
        
        result = benchmark(import_all_modules)
        assert result is None
    
    @pytest.mark.benchmark
    def test_calculator_creation_benchmark(self, benchmark):
        """Benchmark de creación de calculadora."""
        from config.settings import PreconvergenceConfig
        
        def create_calculator():
            config = PreconvergenceConfig()
            from core.calculator import DFTCalculator
            return DFTCalculator(config)
        
        calc = benchmark(create_calculator)
        assert calc is not None
    
    @pytest.mark.benchmark
    @memory_profiler.profile
    def test_memory_usage_during_pipeline(self):
        """Test de uso de memoria durante pipeline."""
        from config.settings import PreconvergenceConfig
        from workflow.pipeline import PreconvergencePipeline
        
        # Monitor de memoria
        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=0.1)
        
        try:
            # Crear pipeline
            config = PreconvergenceConfig()
            pipeline = PreconvergencePipeline(config)
            
            # Simular ejecución breve (sin PySCF real)
            import asyncio
            
            async def mock_execute():
                return await asyncio.sleep(0.1)
            
            # Ejecutar
            asyncio.run(mock_execute())
            
            # Verificar métricas
            summary = monitor.get_performance_summary()
            assert summary['avg_memory_mb'] < 1000  # Menos de 1GB
            
        finally:
            monitor.stop_monitoring()
    
    @pytest.mark.benchmark
    def test_configuration_loading_benchmark(self, benchmark):
        """Benchmark de carga de configuración."""
        from config.settings import PreconvergenceConfig
        
        def load_config():
            config = PreconvergenceConfig()
            return config.to_dict()
        
        result = benchmark(load_config)
        assert 'lattice_constant' in result
```

### 3. Implementación de Estrategias de Caché

#### 3.1 Cache de Resultados de Cálculo
```python
# src/utils/calculation_cache.py
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Dict
import functools
import time

class CalculationCache:
    """Cache para resultados de cálculos DFT."""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 1.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache_index_path = cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Carga índice de cache."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Guarda índice de cache."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Genera clave de cache basada en parámetros."""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Obtiene path de archivo de cache."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        """Obtiene resultado del cache."""
        key = self._generate_key(*args, **kwargs)
        
        if key in self.index:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Actualizar timestamp de acceso
                    self.index[key]['last_accessed'] = time.time()
                    self._save_index()
                    
                    return result
                except Exception as e:
                    print(f"Error reading cache for key {key}: {e}")
        
        return None
    
    def put(self, result: Any, *args, **kwargs):
        """Guarda resultado en cache."""
        key = self._generate_key(*args, **kwargs)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Actualizar índice
            self.index[key] = {
                'created': time.time(),
                'last_accessed': time.time(),
                'size_bytes': cache_path.stat().st_size
            }
            
            # Verificar límites de tamaño
            self._enforce_size_limit()
            self._save_index()
            
        except Exception as e:
            print(f"Error writing cache for key {key}: {e}")
    
    def _enforce_size_limit(self):
        """Enforce tamaño máximo de cache."""
        total_size = sum(info['size_bytes'] for info in self.index.values())
        
        if total_size > self.max_size_bytes:
            # Ordenar por último acceso
            sorted_items = sorted(
                self.index.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            # Eliminar elementos más antiguos
            for key, info in sorted_items:
                if total_size <= self.max_size_bytes * 0.8:  # 80% del límite
                    break
                
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                    total_size -= info['size_bytes']
                    del self.index[key]

# Decorador para cache automático
def cached_calculation(cache: CalculationCache):
    """Decorador para cache automático de cálculos."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Intentar obtener del cache
            result = cache.get(*args, **kwargs)
            if result is not None:
                return result
            
            # Ejecutar cálculo y cachear
            result = func(*args, **kwargs)
            cache.put(result, *args, **kwargs)
            
            return result
        return wrapper
    return decorator

# Uso en calculadora
class DFTCalculator:
    def __init__(self, config: PreconvergenceConfig):
        self.config = config
        self.cache = CalculationCache(config.output_dir / "cache")
    
    @cached_calculation(cache=None)  # Se inicializará después
    async def calculate_energy(self, cell_params) -> EnergyResult:
        # Cálculo real aquí
        pass
```

#### 3.2 Cache de Configuraciones
```python
# src/utils/config_cache.py
"""Cache para configuraciones."""

import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.settings import PreconvergenceConfig

class ConfigCache:
    """Cache de configuraciones para evitar recargas."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_cache: Dict[str, PreconvergenceConfig] = {}
    
    def get_config_hash(self, config_data: Dict) -> str:
        """Genera hash de configuración."""
        config_string = yaml.dump(config_data, sort_keys=True)
        return hashlib.md5(config_string.encode()).hexdigest()
    
    def get_cached_config(self, config_path: Path) -> Optional[PreconvergenceConfig]:
        """Obtiene configuración cacheada."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            config_hash = self.get_config_hash(config_data)
            
            if config_hash in self.config_cache:
                return self.config_cache[config_hash]
            
            # Cargar y cachear
            config = PreconvergenceConfig.load_from_file(config_path)
            self.config_cache[config_hash] = config
            
            return config
            
        except Exception as e:
            print(f"Error loading cached config: {e}")
            return None
    
    def clear_cache(self):
        """Limpia cache de configuraciones."""
        self.config_cache.clear()
```

---

## 📊 MÉTRICAS Y SEGUIMIENTO

### KPIs de Rendimiento
- **Tiempo de Import**: < 500ms para módulos core
- **Memoria Base**: < 200MB sin cálculos
- **Tiempo de Inicialización**: < 2s para pipeline vacío
- **Throughput**: > 10 configuraciones/segundo en cache
- **Cache Hit Rate**: > 80% para cálculos similares

### Herramientas de Monitoreo
- **Performance Monitor**: Métricas en tiempo real
- **Benchmark Suite**: Tests automatizados de rendimiento
- **Memory Profiler**: Análisis detallado de uso de memoria
- **Cache Statistics**: Hit/miss rates y tamaños

### Reportes Automatizados
```python
# src/utils/performance_reporter.py
import json
from datetime import datetime
from pathlib import Path

class PerformanceReporter:
    """Generador de reportes de rendimiento."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_daily_report(self, monitor: PerformanceMonitor):
        """Genera reporte diario de rendimiento."""
        timestamp = datetime.now().strftime("%Y%m%d")
        report_path = self.output_dir / f"performance_report_{timestamp}.json"
        
        summary = monitor.get_performance_summary()
        current_metrics = monitor.get_current_metrics()
        
        report = {
            'timestamp': timestamp,
            'summary': summary,
            'current_metrics': vars(current_metrics) if current_metrics else None,
            'recommendations': self._generate_recommendations(summary)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def _generate_recommendations(self, summary: Dict) -> list:
        """Genera recomendaciones basadas en métricas."""
        recommendations = []
        
        if summary.get('avg_cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected. Consider reducing max_workers.")
        
        if summary.get('avg_memory_mb', 0) > 1000:
            recommendations.append("High memory usage. Consider implementing more aggressive caching.")
        
        return recommendations
```

---

## 🎯 OBJETIVOS DE ÉXITO

### Criterios de Aceptación
1. ✅ **Import Tests**: 100% de módulos se importan sin errores
2. ✅ **Pipeline Tests**: Pipeline completo ejecuta sin fallos
3. ✅ **Performance**: Tiempo de inicialización < 2 segundos
4. ✅ **Memory**: Uso de memoria base < 200MB
5. ✅ **Cache**: Hit rate > 80% para cálculos similares
6. ✅ **Stability**: 0 errores en 100 ejecuciones consecutivas

### Métricas de Éxito
- **Import Integrity**: 100% passing
- **Test Coverage**: > 90%
- **Performance Improvement**: 50% faster than baseline
- **Memory Efficiency**: 30% reduction in peak usage
- **Cache Efficiency**: > 80% hit rate

Este plan de acción detallado proporcionará una base sólida para optimizar y estabilizar el proyecto Preconvergencia-GaAs, asegurando un rendimiento óptimo y mantenibilidad a largo plazo.