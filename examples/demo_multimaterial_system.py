#!/usr/bin/env python3
# examples/demo_multimaterial_system.py
"""
Demostración completa del sistema de preconvergencia multimaterial.

Este ejemplo muestra cómo usar el pipeline multimaterial para ejecutar
cálculos DFT de preconvergencia en múltiples semiconductores de forma eficiente.

Ejecuta una demostración con semiconductores comunes, ejecuta el análisis
completo y genera reportes detallados.
"""

import asyncio
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow.multi_material_pipeline import (
    MultiMaterialPipeline,
    run_common_semiconductors_campaign,
    CampaignResult
)
from analysis.multi_material_analysis import MultiMaterialAnalyzer
from core.multi_material_config import create_common_semiconductors_config
from utils.logging import setup_logging


async def demo_basic_usage():
    """Demuestra el uso básico del sistema multimaterial."""
    print("🚀 DEMO: Uso Básico del Sistema Multimaterial")
    print("=" * 50)
    
    # Crear pipeline con semiconductores comunes
    pipeline = MultiMaterialPipeline()
    
    # Agregar algunos materiales específicos
    materials = ['GaAs', 'GaN', 'InP', 'ZnS', 'CdSe']
    pipeline.add_materials_from_list(materials)
    
    print(f"📋 Materiales configurados: {materials}")
    print(f"⚙️  Configuración paralela: {pipeline.config.parallel_materials}")
    print(f"🔧 Workers configurados: {pipeline.config.max_concurrent_materials}")
    
    # Validar materiales
    validation = pipeline.validate_materials()
    print(f"✅ Materiales válidos: {validation['total_valid']}")
    print(f"❌ Materiales inválidos: {validation['total_invalid']}")
    
    if validation['total_invalid'] > 0:
        print(f"   Errores: {validation['invalid_materials']}")
    
    return pipeline


async def demo_campaign_execution():
    """Demuestra la ejecución de una campaña completa."""
    print("\n🔬 DEMO: Ejecución de Campaña Completa")
    print("=" * 50)
    
    # Seleccionar pocos materiales para demo rápida
    demo_materials = ['GaAs', 'GaN']
    
    print(f"🎯 Ejecutando campaña con: {demo_materials}")
    print("⚙️  Configuración: Paralela, 2 workers")
    
    # Ejecutar campaña con configuración paralela
    result = await run_common_semiconductors_campaign(
        materials=demo_materials,
        parallel=True,
        max_workers=2
    )
    
    print(f"\n📊 Resultados de la campaña:")
    print(f"   • Materiales ejecutados: {result.materials_executed}")
    print(f"   • Materiales exitosos: {result.materials_successful}")
    print(f"   • Materiales fallidos: {result.materials_failed}")
    print(f"   • Tasa de éxito: {result.success_rate:.1f}%")
    print(f"   • Tiempo total: {result.total_execution_time:.2f}s")
    print(f"   • Tiempo promedio: {result.average_execution_time:.2f}s")
    
    return result


async def demo_analysis():
    """Demuestra el sistema de análisis de resultados."""
    print("\n📊 DEMO: Análisis de Resultados")
    print("=" * 50)
    
    # Usar datos simulados para la demo si no hay resultados reales
    print("🔄 Creando datos de demostración...")
    
    # Crear resultado simulado para la demo
    from workflow.multi_material_pipeline import MaterialExecutionResult, CampaignResult
    from core.multi_material_config import MultiMaterialConfig
    
    # Simular resultados de materiales
    demo_results = []
    materials = ['GaAs', 'GaN', 'InP', 'ZnS', 'CdSe']
    
    for i, material in enumerate(materials):
        result = MaterialExecutionResult(
            formula=material,
            success=True,
            execution_time=10.0 + i * 5.0,  # Tiempo simulado
            stages_completed=['cutoff', 'kmesh', 'lattice'],
            optimal_cutoff=400 + i * 50,
            optimal_kmesh=(8, 8, 8),
            optimal_lattice_constant=5.4 + i * 0.2
        )
        demo_results.append(result)
    
    config = create_common_semiconductors_config()
    
    campaign_result = CampaignResult(
        materials_executed=len(materials),
        materials_successful=len(materials),
        materials_failed=0,
        total_execution_time=sum(r.execution_time for r in demo_results),
        individual_results=demo_results,
        campaign_config=config
    )
    
    print(f"📈 Ejecutando análisis de {len(materials)} materiales...")
    
    # Ejecutar análisis
    analyzer = MultiMaterialAnalyzer(enable_visualizations=True)
    analysis_report = analyzer.analyze_campaign_results(
        campaign_result, 
        output_dir=Path("demo_analysis_results")
    )
    
    # Mostrar resumen ejecutivo
    summary = analysis_report.get_executive_summary()
    print(f"\n🎯 Resumen Ejecutivo:")
    print(f"   • Materiales procesados: {summary['campaign_overview']['total_materials']}")
    print(f"   • Tasa de éxito: {summary['campaign_overview']['success_rate']:.1f}%")
    print(f"   • Material más rápido: {summary['key_findings'].get('fastest_material', 'N/A')}")
    print(f"   • Rango de cutoffs: {summary['key_findings'].get('optimal_cutoff_range', 'N/A')}")
    
    print(f"\n💡 Recomendaciones:")
    for i, rec in enumerate(analysis_report.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📁 Análisis completo guardado en: demo_analysis_results/")
    
    return analysis_report


async def demo_configuration():
    """Demuestra las diferentes opciones de configuración."""
    print("\n⚙️ DEMO: Opciones de Configuración")
    print("=" * 50)
    
    # Configuración para semiconductores III-V
    config_iii_v = create_common_semiconductors_config()
    materials_iii_v = [m.formula for m in config_iii_v.get_materials_by_type(
        __import__('src.models.semiconductor_database', fromlist=['SemiconductorType']).SemiconductorType.III_V
    )]
    print(f"🔬 Semiconductores III-V disponibles: {materials_iii_v}")
    
    # Configuración para ejecución secuencial
    pipeline_seq = MultiMaterialPipeline()
    pipeline_seq.enable_parallel_execution(False)
    print(f"🔄 Modo secuencial habilitado")
    
    # Configuración para paralelización
    pipeline_par = MultiMaterialPipeline()
    pipeline_par.enable_parallel_execution(True)
    pipeline_par.set_parallel_workers(6)
    print(f"🚀 Paralelización: {pipeline_par.config.max_concurrent_materials} workers")
    
    # Configuración personalizada
    from core.multi_material_config import MultiMaterialConfig
    custom_config = MultiMaterialConfig()
    custom_config.add_material('GaAs', priority=10)
    custom_config.add_material('GaN', priority=9)
    custom_config.sort_by_priority()
    
    print(f"⚡ Configuración de prioridades:")
    for material in custom_config.materials:
        print(f"   • {material.formula}: prioridad {material.priority}")
    
    return True


async def demo_integration():
    """Demuestra la integración con el sistema existente."""
    print("\n🔗 DEMO: Integración con Sistema Existente")
    print("=" * 50)
    
    try:
        from workflow.pipeline import (
            is_multi_material_available,
            show_multi_material_capabilities,
            validate_multi_material_setup,
            run_single_material_pipeline
        )
        
        # Verificar disponibilidad
        available = is_multi_material_available()
        print(f"✅ Sistema multimaterial disponible: {available}")
        
        if available:
            # Mostrar capacidades
            show_multi_material_capabilities()
            
            # Validar setup
            validation = validate_multi_material_setup()
            print(f"\n🔍 Validación del sistema:")
            print(f"   • Dependencias OK: {validation['dependencies_ok']}")
            print(f"   • Warnings: {len(validation['warnings'])}")
            print(f"   • Errores: {len(validation['errors'])}")
            
            if validation['warnings']:
                print(f"   ⚠️  Warnings:")
                for warning in validation['warnings']:
                    print(f"      - {warning}")
        else:
            print("❌ Sistema multimaterial no disponible")
            print("   Usando pipeline individual...")
    
    except Exception as e:
        print(f"⚠️  Error en demo de integración: {e}")
    
    return True


async def run_full_demo():
    """Ejecuta la demostración completa del sistema."""
    print("🌟 DEMOSTRACIÓN COMPLETA DEL SISTEMA MULTIMATERIAL")
    print("=" * 60)
    print("Este demo muestra todas las capacidades del sistema de")
    print("preconvergencia multimaterial implementado.")
    print("=" * 60)
    
    # Configurar logging
    setup_logging(level='INFO')
    
    try:
        # 1. Demo de uso básico
        pipeline = await demo_basic_usage()
        
        # 2. Demo de configuración
        await demo_configuration()
        
        # 3. Demo de ejecución de campaña
        result = await demo_campaign_execution()
        
        # 4. Demo de análisis
        await demo_analysis()
        
        # 5. Demo de integración
        await demo_integration()
        
        print("\n" + "=" * 60)
        print("🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        print("El sistema multimaterial está funcionando correctamente.")
        print("✅ Funcionalidades demostradas:")
        print("   • Configuración de materiales múltiples")
        print("   • Ejecución paralela/secuencial")
        print("   • Análisis de resultados")
        print("   • Reportes y visualizaciones")
        print("   • Integración con sistema existente")
        
        print("\n📁 Archivos generados durante la demo:")
        print("   • demo_analysis_results/ - Análisis detallado")
        print("   • Log files - Registros de ejecución")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_quick_test():
    """Prueba rápida del sistema."""
    print("⚡ PRUEBA RÁPIDA DEL SISTEMA")
    print("=" * 40)
    
    try:
        # Importar componentes principales
        from workflow.multi_material_pipeline import MultiMaterialPipeline
        from analysis.multi_material_analysis import MultiMaterialAnalyzer
        from core.multi_material_config import create_common_semiconductors_config
        
        print("✅ Imports exitosos")
        
        # Crear pipeline
        pipeline = MultiMaterialPipeline()
        print("✅ Pipeline creado")
        
        # Verificar configuración
        config = create_common_semiconductors_config()
        print(f"✅ Configuración cargada: {len(config.materials)} materiales")
        
        # Verificar análisis
        analyzer = MultiMaterialAnalyzer()
        print("✅ Analizador inicializado")
        
        print("\n🎯 Sistema funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba rápida: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo del sistema multimaterial")
    parser.add_argument('--quick', action='store_true', help='Ejecutar prueba rápida')
    parser.add_argument('--full', action='store_true', help='Ejecutar demo completo')
    
    args = parser.parse_args()
    
    if args.quick:
        success = demo_quick_test()
    elif args.full:
        success = asyncio.run(run_full_demo())
    else:
        # Ejecutar demo completo por defecto
        success = asyncio.run(run_full_demo())
    
    sys.exit(0 if success else 1)
