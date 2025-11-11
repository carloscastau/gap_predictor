#!/usr/bin/env python3
"""
validacion_final.py - Script de validación final del proyecto Preconvergencia-GaAs
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test de imports de todos los módulos principales."""
    print("🔍 Test 1: Verificando imports de módulos...")
    
    modulos_test = [
        "config.settings",
        "core.calculator", 
        "core.optimizer",
        "core.parallel",
        "workflow.pipeline",
        "workflow.checkpoint",
        "utils.logging",
        "analysis.statistics",
        "visualization.plots"
    ]
    
    exitosos = 0
    errores = []
    
    for modulo in modulos_test:
        try:
            __import__(modulo)
            print(f"  ✅ {modulo}")
            exitosos += 1
        except Exception as e:
            print(f"  ❌ {modulo}: {e}")
            errores.append(f"{modulo}: {e}")
    
    print(f"\n📊 Resultado: {exitosos}/{len(modulos_test)} módulos importados correctamente")
    return exitosos == len(modulos_test), errores

def test_configuration():
    """Test de configuración del sistema."""
    print("\n🔧 Test 2: Verificando configuración del sistema...")
    
    try:
        from config.settings import PreconvergenceConfig, get_default_config, get_fast_config
        
        # Test configuración por defecto
        config_default = get_default_config()
        print(f"  ✅ Configuración por defecto: a₀ = {config_default.lattice_constant} Å")
        
        # Test configuración rápida
        config_fast = get_fast_config()
        print(f"  ✅ Configuración rápida: {len(config_fast.cutoff_list)} cutoffs")
        
        # Test validación
        config_custom = PreconvergenceConfig(
            lattice_constant=5.653,
            cutoff_list=[80, 120, 160],
            kmesh_list=[(2,2,2), (4,4,4)]
        )
        print(f"  ✅ Configuración personalizada válida")
        
        return True, []
        
    except Exception as e:
        print(f"  ❌ Error en configuración: {e}")
        return False, [str(e)]

def test_core_classes():
    """Test de clases principales del core."""
    print("\n⚡ Test 3: Verificando clases del core...")
    
    try:
        from config.settings import PreconvergenceConfig
        from core.calculator import DFTCalculator, CellParameters
        from core.optimizer import LatticeOptimizer, ConvergenceAnalyzer
        from core.parallel import MemoryMonitor, TaskScheduler
        
        config = PreconvergenceConfig()
        
        # Test calculadora
        calc = DFTCalculator(config)
        print(f"  ✅ DFTCalculator creado")
        
        # Test parámetros de celda
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
        print(f"  ✅ CellParameters: memoria estimada = {cell_params.estimated_memory:.1f} MB")
        
        # Test optimizador
        opt = LatticeOptimizer(config)
        print(f"  ✅ LatticeOptimizer creado")
        
        # Test analizador de convergencia
        analyzer = ConvergenceAnalyzer(config)
        print(f"  ✅ ConvergenceAnalyzer creado")
        
        # Test monitor de memoria
        monitor = MemoryMonitor()
        print(f"  ✅ MemoryMonitor creado")
        
        # Test scheduler
        scheduler = TaskScheduler(config)
        print(f"  ✅ TaskScheduler creado")
        
        return True, []
        
    except Exception as e:
        print(f"  ❌ Error en clases core: {e}")
        traceback.print_exc()
        return False, [str(e)]

def test_pipeline_structure():
    """Test de estructura del pipeline."""
    print("\n🔄 Test 4: Verificando estructura del pipeline...")
    
    try:
        from config.settings import PreconvergenceConfig
        from workflow.pipeline import PreconvergencePipeline, PreconvergenceConfig
        from workflow.checkpoint import CheckpointManager
        
        config = PreconvergenceConfig()
        pipeline = PreconvergencePipeline(config)
        
        print(f"  ✅ PreconvergencePipeline creado")
        print(f"  ✅ Stages disponibles: {list(pipeline.stages.keys())}")
        
        # Test checkpoint manager
        checkpoint_manager = pipeline.checkpoint_manager
        print(f"  ✅ CheckpointManager inicializado")
        
        return True, []
        
    except Exception as e:
        print(f"  ❌ Error en pipeline: {e}")
        traceback.print_exc()
        return False, [str(e)]

def test_dependencies():
    """Test de dependencias científicas."""
    print("\n🔬 Test 5: Verificando dependencias científicas...")
    
    dependencias = [
        ("numpy", "NumPy - Cálculos numéricos"),
        ("scipy", "SciPy - Algoritmos científicos"), 
        ("pandas", "Pandas - Manejo de datos"),
        ("matplotlib", "Matplotlib - Visualización"),
        ("pymatgen", "PyMatGen - Cristalografía"),
        ("spglib", "SPGLIB - Análisis de cristales")
    ]
    
    exitosos = 0
    
    for dep_name, desc in dependencias:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_name}: {desc}")
            exitosos += 1
        except ImportError:
            print(f"  ❌ {dep_name}: No disponible")
        except Exception as e:
            print(f"  ⚠️  {dep_name}: Error - {e}")
    
    print(f"\n📊 Dependencias: {exitosos}/{len(dependencias)} disponibles")
    return exitosos >= 4, []  # Al menos 4 dependencias principales

def test_performance():
    """Test básico de rendimiento."""
    print("\n⚡ Test 6: Verificando rendimiento básico...")
    
    try:
        from config.settings import PreconvergenceConfig
        from core.calculator import DFTCalculator
        import time
        
        start_time = time.time()
        
        # Crear múltiples calculadoras (test de performance)
        config = PreconvergenceConfig()
        calculators = []
        for i in range(10):
            calc = DFTCalculator(config)
            calculators.append(calc)
        
        creation_time = time.time() - start_time
        
        print(f"  ✅ Creación de 10 calculadoras: {creation_time:.3f}s")
        print(f"  ✅ Tiempo promedio por calculadora: {creation_time/10:.3f}s")
        
        # Test de memoria estimada
        total_memory = sum(c.calculator.estimated_memory for c in calculators)
        print(f"  ✅ Memoria total estimada: {total_memory:.1f} MB")
        
        return True, []
        
    except Exception as e:
        print(f"  ❌ Error en performance: {e}")
        return False, [str(e)]

def main():
    """Ejecuta todos los tests de validación."""
    print("=" * 70)
    print("🚀 VALIDACIÓN FINAL DEL PROYECTO PRECONVERGENCIA-GAAS")
    print("=" * 70)
    print(f"📅 Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print()
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        print(f"📁 Directorio src agregado al path: {src_path}")
    else:
        print(f"⚠️  Directorio src no encontrado en: {src_path}")
    
    print()
    
    # Ejecutar tests
    tests = [
        ("Imports de Módulos", test_imports),
        ("Configuración del Sistema", test_configuration),
        ("Clases del Core", test_core_classes),
        ("Estructura del Pipeline", test_pipeline_structure),
        ("Dependencias Científicas", test_dependencies),
        ("Rendimiento Básico", test_performance)
    ]
    
    resultados = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"EJECUTANDO: {test_name}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            success, errors = test_func()
            duration = time.time() - start_time
            
            resultados.append({
                'test': test_name,
                'success': success,
                'duration': duration,
                'errors': errors
            })
            
            status = "✅ PASÓ" if success else "❌ FALLÓ"
            print(f"\n🏁 {test_name}: {status} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR CRÍTICO - {e}")
            resultados.append({
                'test': test_name,
                'success': False,
                'duration': 0,
                'errors': [f"Critical error: {e}"]
            })
    
    # Resumen final
    print(f"\n\n{'='*70}")
    print("📊 RESUMEN FINAL DE VALIDACIÓN")
    print(f"{'='*70}")
    
    tests_passed = sum(1 for r in resultados if r['success'])
    total_tests = len(resultados)
    
    for resultado in resultados:
        status = "✅" if resultado['success'] else "❌"
        print(f"{status} {resultado['test']}: {resultado['duration']:.2f}s")
        if resultado['errors']:
            for error in resultado['errors'][:3]:  # Mostrar solo primeros 3 errores
                print(f"    • {error}")
    
    print(f"\n📈 PUNTUACIÓN FINAL: {tests_passed}/{total_tests} tests pasaron")
    print(f"📊 Porcentaje de éxito: {(tests_passed/total_tests)*100:.1f}%")
    
    # Determinar estado general
    if tests_passed == total_tests:
        estado = "🎉 EXCELENTE - Proyecto completamente funcional"
    elif tests_passed >= total_tests * 0.8:
        estado = "✅ BUENO - Proyecto mayormente funcional"
    elif tests_passed >= total_tests * 0.6:
        estado = "⚠️  ACEPTABLE - Proyecto parcialmente funcional"
    else:
        estado = "❌ PROBLEMAS - Proyecto requiere correcciones"
    
    print(f"\n🎯 ESTADO GENERAL: {estado}")
    
    print(f"\n{'='*70}")
    print("🚀 VALIDACIÓN COMPLETADA")
    print(f"{'='*70}")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Error crítico durante validación: {e}")
        traceback.print_exc()
        exit(1)