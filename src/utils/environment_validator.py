# src/utils/environment_validator.py
"""
Validador completo de entorno para producción Preconvergencia-GaAs v2.0
Implementa verificaciones críticas antes del despliegue
"""

import subprocess
import sys
import time
import psutil
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class ValidationResult:
    """Resultado de validación de componente."""
    component: str
    status: str  # 'PASS', 'FAIL', 'WARN'
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0

class EnvironmentValidator:
    """Validador completo de entorno para producción."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def log(self, message: str):
        """Log con formato consistente."""
        if self.verbose:
            timestamp = time.strftime('%H:%M:%S')
            print(f"[{timestamp}] {message}")
    
    def validate_compilation_tools(self) -> ValidationResult:
        """Valida herramientas de compilación."""
        self.log("🔧 Validando herramientas de compilación...")
        start_time = time.time()
        
        tools = {
            'gfortran': ['gfortran', '--version'],
            'gcc': ['gcc', '--version'],
            'cmake': ['cmake', '--version'],
            'make': ['make', '--version']
        }
        
        missing_tools = []
        available_tools = {}
        
        for tool, command in tools.items():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    available_tools[tool] = version_line
                    self.log(f"  ✅ {tool}: {version_line}")
                else:
                    missing_tools.append(tool)
                    self.log(f"  ❌ {tool}: Failed")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(tool)
                self.log(f"  ❌ {tool}: Not found")
        
        execution_time = time.time() - start_time
        
        if missing_tools:
            return ValidationResult(
                component='compilation_tools',
                status='FAIL',
                message=f"Missing tools: {', '.join(missing_tools)}",
                details={'missing': missing_tools, 'available': available_tools},
                execution_time=execution_time
            )
        
        return ValidationResult(
            component='compilation_tools',
            status='PASS',
            message="All compilation tools available",
            details=available_tools,
            execution_time=execution_time
        )
    
    def validate_system_libraries(self) -> ValidationResult:
        """Valida librerías del sistema."""
        self.log("📚 Validando librerías del sistema...")
        start_time = time.time()
        
        # Verificar librerías BLAS/LAPACK
        blas_libraries = [
            'libblas.so',
            'liblapack.so', 
            'libopenblas.so',
            'libmkl.so'
        ]
        
        found_libraries = []
        missing_libraries = []
        
        for lib in blas_libraries:
            try:
                result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
                if lib in result.stdout:
                    found_libraries.append(lib)
                    self.log(f"  ✅ {lib} found")
                else:
                    missing_libraries.append(lib)
                    self.log(f"  ⚠️  {lib} not found")
            except Exception:
                missing_libraries.append(lib)
                self.log(f"  ❌ {lib}: Check failed")
        
        execution_time = time.time() - start_time
        
        if len(found_libraries) == 0:
            return ValidationResult(
                component='system_libraries',
                status='FAIL',
                message="No BLAS/LAPACK libraries found",
                details={'found': found_libraries, 'missing': missing_libraries},
                execution_time=execution_time
            )
        
        if len(missing_libraries) > 0:
            return ValidationResult(
                component='system_libraries',
                status='WARN',
                message=f"Partial BLAS/LAPACK support: {len(found_libraries)} found",
                details={'found': found_libraries, 'missing': missing_libraries},
                execution_time=execution_time
            )
        
        return ValidationResult(
            component='system_libraries',
            status='PASS',
            message="All required system libraries found",
            details={'found': found_libraries},
            execution_time=execution_time
        )
    
    def validate_pyscf_installation(self) -> ValidationResult:
        """Valida instalación funcional de PySCF."""
        self.log("⚛️  Validando instalación de PySCF...")
        start_time = time.time()
        
        try:
            import pyscf
            from pyscf.pbc import gto, dft
            
            # Test de construcción de celda básica
            cell = gto.Cell()
            cell.atom = 'C 0 0 0'
            cell.basis = 'gth-dzvp'
            cell.a = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
            cell.build()
            
            # Test de configuración de memoria
            import pyscf.lib
            max_memory = pyscf.lib.param.MAX_MEMORY
            
            # Test de funcionalidad DFT básica
            kpts = cell.make_kpts((2, 2, 2))
            kmf = dft.KRKS(cell, kpts=kpts)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                component='pyscf_installation',
                status='PASS',
                message=f"PySCF fully functional (version: {pyscf.__version__})",
                details={
                    'version': pyscf.__version__,
                    'max_memory_gb': max_memory / (1024**3),
                    'cell_build_successful': True,
                    'dft_setup_successful': True
                },
                execution_time=execution_time
            )
            
        except ImportError as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                component='pyscf_installation',
                status='FAIL',
                message=f"PySCF import failed: {e}",
                details={'error': str(e)},
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                component='pyscf_installation',
                status='WARN',
                message=f"PySCF install incomplete: {e}",
                details={'error': str(e), 'partial_functionality': True},
                execution_time=execution_time
            )
    
    def validate_scientific_libraries(self) -> ValidationResult:
        """Valida librerías científicas principales."""
        self.log("🔬 Validando librerías científicas...")
        start_time = time.time()
        
        libraries = {
            'numpy': 'numpy',
            'scipy': 'scipy',
            'matplotlib': 'matplotlib',
            'pymatgen': 'pymatgen',
            'spglib': 'spglib'
        }
        
        results = {}
        failed_imports = []
        
        for name, module in libraries.items():
            try:
                exec(f"import {module}")
                version = eval(f"{module}.__version__")
                results[name] = version
                self.log(f"  ✅ {name}: {version}")
            except ImportError:
                failed_imports.append(name)
                self.log(f"  ❌ {name}: Not available")
            except Exception as e:
                failed_imports.append(name)
                self.log(f"  ⚠️  {name}: Error - {e}")
        
        execution_time = time.time() - start_time
        
        if failed_imports:
            return ValidationResult(
                component='scientific_libraries',
                status='WARN' if len(failed_imports) < len(libraries) // 2 else 'FAIL',
                message=f"Partial scientific library support: {len(results)}/{len(libraries)} available",
                details={'available': results, 'failed': failed_imports},
                execution_time=execution_time
            )
        
        return ValidationResult(
            component='scientific_libraries',
            status='PASS',
            message="All scientific libraries available",
            details=results,
            execution_time=execution_time
        )
    
    def validate_blas_performance(self) -> ValidationResult:
        """Valida configuración y performance de librerías BLAS."""
        self.log("⚡ Validando performance de BLAS...")
        start_time = time.time()
        
        try:
            # Test de performance BLAS
            size = 1000
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            import time
            start_compute = time.time()
            C = A @ B
            end_compute = time.time()
            
            computation_time = end_compute - start_compute
            
            execution_time = time.time() - start_time
            
            # Verificar que no es implementación Python pura
            if computation_time < 2.0:  # Menos de 2 segundos para 1000x1000
                return ValidationResult(
                    component='blas_performance',
                    status='PASS',
                    message="BLAS libraries properly configured for high performance",
                    details={
                        'computation_time': computation_time,
                        'estimated_gflops': (2 * size**3) / (computation_time * 1e9)
                    },
                    execution_time=execution_time
                )
            elif computation_time < 10.0:
                return ValidationResult(
                    component='blas_performance',
                    status='WARN',
                    message="BLAS configured but performance suboptimal",
                    details={
                        'computation_time': computation_time,
                        'note': 'May indicate single-threaded or inefficient BLAS'
                    },
                    execution_time=execution_time
                )
            else:
                return ValidationResult(
                    component='blas_performance',
                    status='FAIL',
                    message="BLAS performance severely degraded - possible Python fallback",
                    details={'computation_time': computation_time},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                component='blas_performance',
                status='FAIL',
                message=f"BLAS performance test failed: {e}",
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def validate_memory_requirements(self, config) -> ValidationResult:
        """Valida requisitos de memoria para configuración dada."""
        self.log("💾 Validando requisitos de memoria...")
        start_time = time.time()
        
        # Calcular memoria estimada basada en configuración
        max_kmesh = max(config.kmesh_list)
        nkpts = max_kmesh[0] * max_kmesh[1] * max_kmesh[2]
        estimated_memory_gb = 200 + nkpts * 50 / 1024  # MB a GB
        
        # Factores de seguridad
        safety_factors = {
            'base_overhead': 1.2,  # 20% overhead
            'safety_margin': 1.5   # 50% safety margin
        }
        
        required_memory_gb = estimated_memory_gb * safety_factors['base_overhead'] * safety_factors['safety_margin']
        
        # Verificar memoria disponible
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        total_memory_gb = memory.total / (1024**3)
        
        execution_time = time.time() - start_time
        
        if available_memory_gb >= required_memory_gb:
            return ValidationResult(
                component='memory_requirements',
                status='PASS',
                message="Sufficient memory available for configuration",
                details={
                    'estimated_required_gb': required_memory_gb,
                    'estimated_base_gb': estimated_memory_gb,
                    'available_gb': available_memory_gb,
                    'total_gb': total_memory_gb,
                    'margin_gb': available_memory_gb - required_memory_gb,
                    'safety_factors': safety_factors
                },
                execution_time=execution_time
            )
        elif available_memory_gb >= estimated_memory_gb:
            return ValidationResult(
                component='memory_requirements',
                status='WARN',
                message="Memory available but close to recommended limits",
                details={
                    'estimated_required_gb': required_memory_gb,
                    'estimated_base_gb': estimated_memory_gb,
                    'available_gb': available_memory_gb,
                    'shortfall_gb': required_memory_gb - available_memory_gb
                },
                execution_time=execution_time
            )
        else:
            return ValidationResult(
                component='memory_requirements',
                status='FAIL',
                message="Insufficient memory for configuration",
                details={
                    'estimated_required_gb': required_memory_gb,
                    'estimated_base_gb': estimated_memory_gb,
                    'available_gb': available_memory_gb,
                    'shortfall_gb': estimated_memory_gb - available_memory_gb
                },
                execution_time=execution_time
            )
    
    def validate_parallelization_setup(self, config) -> ValidationResult:
        """Valida configuración de paralelismo."""
        self.log("🔄 Validando configuración de paralelismo...")
        start_time = time.time()
        
        import multiprocessing
        import os
        
        # Verificar configuración de threads
        omp_threads = os.environ.get('OMP_NUM_THREADS', '1')
        openblas_threads = os.environ.get('OPENBLAS_NUM_THREADS', '1')
        mkl_threads = os.environ.get('MKL_NUM_THREADS', '1')
        
        cpu_count = multiprocessing.cpu_count()
        configured_workers = config.max_workers
        
        # Verificar que la configuración es razonable
        issues = []
        recommendations = []
        
        if configured_workers > cpu_count:
            issues.append(f"Workers ({configured_workers}) exceed CPU count ({cpu_count})")
            recommendations.append(f"Reduce max_workers to {cpu_count // 2}")
        
        if int(omp_threads) > 4:
            issues.append(f"OMP_NUM_THREADS ({omp_threads}) may cause oversubscription")
            recommendations.append("Set OMP_NUM_THREADS to 1-4")
        
        execution_time = time.time() - start_time
        
        if issues:
            return ValidationResult(
                component='parallelization_setup',
                status='WARN',
                message="Parallelization configuration has potential issues",
                details={
                    'cpu_count': cpu_count,
                    'configured_workers': configured_workers,
                    'omp_threads': omp_threads,
                    'openblas_threads': openblas_threads,
                    'mkl_threads': mkl_threads,
                    'issues': issues,
                    'recommendations': recommendations
                },
                execution_time=execution_time
            )
        
        return ValidationResult(
            component='parallelization_setup',
            status='PASS',
            message="Parallelization configuration looks good",
            details={
                'cpu_count': cpu_count,
                'configured_workers': configured_workers,
                'omp_threads': omp_threads,
                'environment_optimal': True
            },
            execution_time=execution_time
        )
    
    def validate_project_structure(self) -> ValidationResult:
        """Valida estructura del proyecto."""
        self.log("📁 Validando estructura del proyecto...")
        start_time = time.time()
        
        # Verificar archivos y directorios críticos
        critical_paths = {
            'src/config/settings.py': 'Configuración principal',
            'src/core/calculator.py': 'Calculadora DFT',
            'src/core/optimizer.py': 'Optimizador',
            'src/workflow/pipeline.py': 'Pipeline principal',
            'src/workflow/checkpoint': 'Sistema de checkpoints',
            'config': 'Directorio de configuraciones'
        }
        
        missing_paths = []
        existing_paths = []
        
        for path, description in critical_paths.items():
            if Path(path).exists():
                existing_paths.append((path, description))
                self.log(f"  ✅ {path}: {description}")
            else:
                missing_paths.append((path, description))
                self.log(f"  ❌ {path}: {description} - MISSING")
        
        execution_time = time.time() - start_time
        
        if missing_paths:
            return ValidationResult(
                component='project_structure',
                status='FAIL',
                message=f"Critical project files missing: {len(missing_paths)}",
                details={'missing': missing_paths, 'existing': existing_paths},
                execution_time=execution_time
            )
        
        return ValidationResult(
            component='project_structure',
            status='PASS',
            message="Project structure is complete",
            details={'checked': len(critical_paths), 'existing': existing_paths},
            execution_time=execution_time
        )
    
    def run_full_validation(self, config) -> Dict[str, Any]:
        """Ejecuta validación completa del entorno."""
        self.log("🚀 Iniciando validación completa del entorno...")
        self.log("=" * 60)
        
        validations = [
            self.validate_compilation_tools,
            self.validate_system_libraries,
            self.validate_scientific_libraries,
            self.validate_pyscf_installation,
            self.validate_blas_performance,
            self.validate_memory_requirements,
            self.validate_parallelization_setup,
            self.validate_project_structure
        ]
        
        # Ejecutar validaciones
        for validation_func in validations:
            try:
                result = validation_func(config)
                self.results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    component=validation_func.__name__,
                    status='ERROR',
                    message=f"Validation failed: {str(e)}",
                    details={'error': str(e)}
                )
                self.results.append(error_result)
                self.log(f"  💥 Validation error: {e}")
        
        # Generar reporte final
        total_time = time.time() - self.start_time
        
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        warnings = sum(1 for r in self.results if r.status == 'WARN')
        errors = sum(1 for r in self.results if r.status == 'ERROR')
        
        # Determinar estado general
        if failed > 0 or errors > 0:
            overall_status = 'NOT_READY'
        elif warnings > 0:
            overall_status = 'READY_WITH_WARNINGS'
        else:
            overall_status = 'READY'
        
        self.log("=" * 60)
        self.log(f"✅ Validación completada en {total_time:.2f} segundos")
        self.log(f"📊 Resumen: {passed} ✅, {warnings} ⚠️, {failed} ❌, {errors} 💥")
        
        return {
            'overall_status': overall_status,
            'total_execution_time': total_time,
            'summary': {
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'errors': errors,
                'total': len(self.results)
            },
            'execution_times': {r.component: r.execution_time for r in self.results},
            'details': [vars(r) for r in self.results]
        }

# Función de utilidad para uso directo
def validate_production_environment(config, verbose: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para validar entorno de producción.
    
    Args:
        config: Configuración del proyecto
        verbose: Si mostrar output detallado
        
    Returns:
        Dict con resultados de validación
    """
    validator = EnvironmentValidator(verbose=verbose)
    return validator.run_full_validation(config)

# Si se ejecuta como script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Agregar src al path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from config.settings import PreconvergenceConfig
        config = PreconvergenceConfig()
        
        print("🔍 Validación de Entorno de Producción - Preconvergencia-GaAs v2.0")
        print("=" * 80)
        
        result = validate_production_environment(config, verbose=True)
        
        print("\n" + "=" * 80)
        print(f"🎯 ESTADO GENERAL: {result['overall_status']}")
        
        if result['overall_status'] == 'READY':
            print("✅ El entorno está listo para producción")
            sys.exit(0)
        elif result['overall_status'] == 'READY_WITH_WARNINGS':
            print("⚠️  El entorno está listo con advertencias")
            sys.exit(0)
        else:
            print("❌ El entorno NO está listo para producción")
            print("\n📋 Detalles de fallos:")
            for detail in result['details']:
                if detail['status'] in ['FAIL', 'ERROR']:
                    print(f"  - {detail['component']}: {detail['message']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Error durante la validación: {e}")
        sys.exit(1)