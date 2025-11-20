#!/usr/bin/env python3
# examples/uso_basico_multimaterial.py
"""
Ejemplo básico de uso del pipeline multimaterial.

Este ejemplo muestra cómo usar el sistema de forma simple y rápida
para ejecutar preconvergencia en múltiples semiconductores.
"""

import asyncio
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Importar componentes principales
from workflow.multi_material_pipeline import run_common_semiconductors_campaign
from analysis.multi_material_analysis import analyze_campaign_quick


async def ejemplo_basico():
    """Ejemplo básico de uso del sistema multimaterial."""
    print("🚀 EJEMPLO BÁSICO - PIPELINE MULTIMATERIAL")
    print("=" * 50)
    
    # 1. Ejecutar campaña con semiconductores comunes
    print("📋 Ejecutando campaña con GaAs y GaN...")
    
    result = await run_common_semiconductors_campaign(
        materials=['GaAs', 'GaN'],  # Solo 2 materiales para demo rápida
        parallel=True,               # Ejecutar en paralelo
        max_workers=2                # 2 workers
    )
    
    # 2. Mostrar resultados básicos
    print(f"\n✅ RESULTADOS:")
    print(f"   • Materiales ejecutados: {result.materials_executed}")
    print(f"   • Exitosos: {result.materials_successful}")
    print(f"   • Fallidos: {result.materials_failed}")
    print(f"   • Tasa de éxito: {result.success_rate:.1f}%")
    print(f"   • Tiempo total: {result.total_execution_time:.2f}s")
    
    # 3. Análisis rápido
    print(f"\n📊 ANÁLISIS RÁPIDO:")
    summary = analyze_campaign_quick(result)
    
    successful_materials = summary['key_findings']['successful_materials']
    fastest_material = summary['key_findings']['fastest_material']
    
    print(f"   • Materiales exitosos: {successful_materials}")
    print(f"   • Material más rápido: {fastest_material}")
    
    return result


async def ejemplo_comparacion():
    """Ejemplo de comparación entre materiales."""
    print("\n🔬 EJEMPLO - COMPARACIÓN DE MATERIALES")
    print("=" * 50)
    
    # Comparar semiconductores III-V vs II-VI
    materials_iii_v = ['GaAs', 'GaN', 'InP']
    materials_ii_vi = ['ZnS', 'ZnSe', 'CdSe']
    
    print(f"🔬 Materiales III-V: {materials_iii_v}")
    result_iii_v = await run_common_semiconductors_campaign(
        materials=materials_iii_v,
        parallel=True,
        max_workers=3
    )
    
    print(f"\n⚗️ Materiales II-VI: {materials_ii_vi}")
    result_ii_vi = await run_common_semiconductors_campaign(
        materials=materials_ii_vi,
        parallel=True,
        max_workers=3
    )
    
    # Comparar resultados
    print(f"\n📊 COMPARACIÓN:")
    print(f"   III-V - Éxito: {result_iii_v.success_rate:.1f}%, "
          f"Tiempo: {result_iii_v.average_execution_time:.1f}s")
    print(f"   II-VI - Éxito: {result_ii_vi.success_rate:.1f}%, "
          f"Tiempo: {result_ii_vi.average_execution_time:.1f}s")
    
    return result_iii_v, result_ii_vi


async def main():
    """Función principal del ejemplo."""
    try:
        # Ejemplo básico
        result = await ejemplo_basico()
        
        # Ejemplo de comparación
        await ejemplo_comparacion()
        
        print(f"\n🎉 EJEMPLOS COMPLETADOS")
        print(f"El pipeline multimaterial funciona correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Ejecutar ejemplo
    success = asyncio.run(main())
    
    if success:
        print(f"\n💡 Para ejecutar campañas más grandes:")
        print(f"   python examples/uso_basico_multimaterial.py")
        print(f"\n💡 Para documentación completa:")
        print(f"   Ver: docs/PIPELINE_MULTIMATERIAL_DOCUMENTACION.md")
        print(f"\n💡 Para script de línea de comandos:")
        print(f"   python scripts/run_preconvergence_campaign.py --help")
    
    sys.exit(0 if success else 1)