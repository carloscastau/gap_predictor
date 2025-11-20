#!/usr/bin/env python3
# scripts/run_preconvergence_campaign.py
"""
Script principal para ejecutar campañas de preconvergencia multimaterial.

Este script proporciona una interfaz de línea de comandos para ejecutar
campañas de preconvergencia DFT para múltiples materiales semiconductores.

Uso:
    python run_preconvergence_campaign.py --type common --materials GaAs,GaN,InP
    python run_preconvergence_campaign.py --type generated --max-materials 5
    python run_preconvergence_campaign.py --type custom --materials ZnS,CdSe --parallel --workers 4
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow.multi_material_pipeline import (
    MultiMaterialPipeline,
    run_common_semiconductors_campaign,
    run_custom_materials_campaign,
    run_generated_materials_campaign,
    CampaignResult
)
from analysis.multi_material_analysis import MultiMaterialAnalyzer, analyze_campaign_quick
from core.multi_material_config import (
    MultiMaterialConfig,
    create_common_semiconductors_config,
    create_iii_v_config,
    create_ii_vi_config
)
from core.material_permutator import (
    MaterialPermutator,
    PermutationFilter,
    MATERIAL_PERMUTATOR
)
from models.semiconductor_database import SemiconductorType, SEMICONDUCTOR_DB
from utils.logging import setup_logging


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Ejecuta campañas de preconvergencia multimaterial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Campaña con semiconductores comunes
  python run_preconvergence_campaign.py --type common

  # Campaña con materiales específicos
  python run_preconvergence_campaign.py --type common --materials GaAs,GaN,InP

  # Campaña con materiales personalizados
  python run_preconvergence_campaign.py --type custom --materials ZnS,CdSe,HgTe

  # Campaña con materiales generados automáticamente
  python run_preconvergence_campaign.py --type generated --max-materials 10

  # Campaña con ejecución paralela
  python run_preconvergence_campaign.py --materials GaAs,GaN --parallel --workers 6

  # Campaña con análisis detallado
  python run_preconvergence_campaign.py --materials GaAs,GaN --analyze --output results/
        """
    )
    
    # Tipo de campaña
    parser.add_argument(
        '--type', '-t',
        choices=['common', 'custom', 'generated'],
        default='common',
        help='Tipo de campaña a ejecutar (default: common)'
    )
    
    # Materiales
    parser.add_argument(
        '--materials', '-m',
        type=str,
        help='Lista de materiales separados por comas (ej: GaAs,GaN,InP)'
    )
    
    # Configuración de ejecución
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Ejecutar materiales en paralelo'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Número de workers paralelos (default: 4)'
    )
    
    # Generación automática
    parser.add_argument(
        '--max-materials',
        type=int,
        default=10,
        help='Máximo número de materiales a generar (default: 10)'
    )
    
    parser.add_argument(
        '--semiconductor-types',
        nargs='+',
        choices=['III_V', 'II_VI'],
        default=['III_V', 'II_VI'],
        help='Tipos de semiconductores a generar (default: ambos)'
    )
    
    # Configuración de análisis
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Ejecutar análisis detallado de resultados'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results_campaign',
        help='Directorio de salida (default: results_campaign)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Salida verbose'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Salida mínima'
    )
    
    # Configuración avanzada
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Stage desde donde reanudar ejecución'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Solo validar materiales sin ejecutar'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Archivo de configuración personalizada (YAML)'
    )
    
    return parser.parse_args()


async def run_campaign(args):
    """Ejecuta la campaña según los argumentos."""
    
    # Configurar logging
    log_level = 'DEBUG' if args.verbose else 'WARNING' if args.quiet else 'INFO'
    setup_logging(level=log_level)
    
    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Iniciando campaña de preconvergencia multimaterial")
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"⚙️  Configuración: {'Paralela' if args.parallel else 'Secuencial'} "
          f"({args.workers} workers)" if args.parallel else "Secuencial")
    print("-" * 60)
    
    # Cargar configuración personalizada si se proporciona
    config = None
    if args.config_file:
        try:
            config = MultiMaterialConfig.load_from_file(Path(args.config_file))
            print(f"📋 Configuración cargada desde: {args.config_file}")
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
            return False
    
    # Ejecutar según el tipo de campaña
    try:
        if args.type == 'common':
            result = await run_common_campaign(args, config, output_dir)
        elif args.type == 'custom':
            result = await run_custom_campaign(args, config, output_dir)
        elif args.type == 'generated':
            result = await run_generated_campaign(args, config, output_dir)
        else:
            print(f"❌ Tipo de campaña no reconocido: {args.type}")
            return False
        
        if result:
            print("\n" + "=" * 60)
            print("🎉 CAMPAÑA COMPLETADA EXITOSAMENTE")
            print(f"✅ Materiales exitosos: {result.materials_successful}")
            print(f"❌ Materiales fallidos: {result.materials_failed}")
            print(f"📊 Tasa de éxito: {result.success_rate:.1f}%")
            print(f"⏱️  Tiempo total: {result.total_execution_time:.2f}s")
            print(f"📈 Tiempo promedio: {result.average_execution_time:.2f}s")
            
            # Guardar resultados
            results_file = output_dir / "campaign_results.json"
            pipeline = MultiMaterialPipeline()
            pipeline.save_campaign_results(result, results_file)
            print(f"💾 Resultados guardados en: {results_file}")
            
            # Ejecutar análisis si se solicita
            if args.analyze:
                await run_analysis(result, output_dir)
            
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ CAMPAÑA FALLÓ")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return False


async def run_common_campaign(args, config, output_dir):
    """Ejecuta campaña con semiconductores comunes."""
    print("🔬 Ejecutando campaña con semiconductores comunes")
    
    # Obtener lista de materiales
    if args.materials:
        materials = [m.strip() for m in args.materials.split(',')]
        print(f"📝 Materiales especificados: {materials}")
    else:
        materials = None
        print("📝 Usando semiconductores comunes predefinidos")
    
    # Ejecutar campaña
    result = await run_common_semiconductors_campaign(
        materials=materials,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Mostrar progreso si es secuencial
    if not args.parallel and not args.quiet:
        print_progress = lambda completed, total: print(f"\r⏳ Progreso: {completed}/{total}", end="")
        # Nota: El progreso secuencial se maneja internamente
        pass
    
    return result


async def run_custom_campaign(args, config, output_dir):
    """Ejecuta campaña con materiales personalizados."""
    print("🛠️  Ejecutando campaña con materiales personalizados")
    
    # Validar materiales especificados
    if not args.materials:
        print("❌ Debe especificar materiales con --materials para campaña custom")
        return None
    
    materials = [m.strip() for m in args.materials.split(',')]
    
    # Validar que existen en la base de datos
    missing_materials = []
    for material in materials:
        if material not in SEMICONDUCTOR_DB.semiconductors:
            missing_materials.append(material)
    
    if missing_materials:
        print(f"⚠️  Materiales no encontrados en base de datos: {missing_materials}")
        print("💡 Continuando con materiales válidos...")
        materials = [m for m in materials if m in SEMICONDUCTOR_DB.semiconductors]
    
    if not materials:
        print("❌ No hay materiales válidos para ejecutar")
        return None
    
    print(f"✅ Materiales validados: {materials}")
    
    # Ejecutar campaña
    result = await run_custom_materials_campaign(
        materials=materials,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    return result


async def run_generated_campaign(args, config, output_dir):
    """Ejecuta campaña con materiales generados automáticamente."""
    print("🤖 Ejecutando campaña con materiales generados automáticamente")
    
    # Configurar tipos de semiconductores
    sem_types = []
    for sem_type_str in args.semiconductor_types:
        if sem_type_str == 'III_V':
            sem_types.append(SemiconductorType.III_V)
        elif sem_type_str == 'II_VI':
            sem_types.append(SemiconductorType.II_VI)
    
    print(f"🔬 Tipos de semiconductores: {[st.value for st in sem_types]}")
    print(f"📊 Máximo de materiales: {args.max_materials}")
    
    # Generar preview de materiales
    preview_materials = []
    for sem_type in sem_types:
        if sem_type == SemiconductorType.III_V:
            result = MATERIAL_PERMUTATOR.generate_iii_v_combinations()
        elif sem_type == SemiconductorType.II_VI:
            result = MATERIAL_PERMUTATOR.generate_ii_vi_combinations()
        else:
            continue
        
        preview_materials.extend([sc.formula for sc in result.filtered_combinations[:args.max_materials//len(sem_types)]])
        
        print(f"📈 Generados {len(result.filtered_combinations)} {sem_type.value}, "
              f"usando {min(len(result.filtered_combinations), args.max_materials//len(sem_types))}")
    
    if len(preview_materials) > args.max_materials:
        preview_materials = preview_materials[:args.max_materials]
    
    print(f"🎯 Total de materiales a procesar: {len(preview_materials)}")
    if len(preview_materials) <= 20:
        print(f"📝 Materiales: {preview_materials}")
    
    # Ejecutar campaña
    result = await run_generated_materials_campaign(
        semiconductor_types=sem_types,
        max_materials=args.max_materials,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    return result


async def run_analysis(campaign_result: CampaignResult, output_dir: Path):
    """Ejecuta análisis detallado de resultados."""
    print("\n" + "=" * 60)
    print("📊 EJECUTANDO ANÁLISIS DETALLADO")
    
    try:
        analyzer = MultiMaterialAnalyzer(enable_visualizations=True)
        analysis_report = analyzer.analyze_campaign_results(campaign_result, output_dir / "analysis")
        
        print("📈 Análisis completado:")
        executive_summary = analysis_report.get_executive_summary()
        
        print(f"🎯 Resumen ejecutivo:")
        key_findings = executive_summary['key_findings']
        print(f"   • Material más rápido: {key_findings.get('fastest_material', 'N/A')}")
        print(f"   • Material más lento: {key_findings.get('slowest_material', 'N/A')}")
        print(f"   • Cutoff más convergente: {key_findings.get('most_convergent_cutoff', 'N/A')}")
        
        if key_findings.get('optimal_cutoff_range'):
            min_cutoff, max_cutoff = key_findings['optimal_cutoff_range']
            print(f"   • Rango de cutoffs: {min_cutoff:.0f} - {max_cutoff:.0f} eV")
        
        print(f"\n💡 Recomendaciones:")
        for i, rec in enumerate(analysis_report.recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📁 Resultados de análisis guardados en: {output_dir / 'analysis'}")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


async def validate_materials(args):
    """Solo valida materiales sin ejecutar."""
    print("🔍 VALIDANDO MATERIALES")
    print("-" * 40)
    
    materials = []
    
    if args.type == 'common' and args.materials:
        materials = [m.strip() for m in args.materials.split(',')]
    elif args.type == 'custom' and args.materials:
        materials = [m.strip() for m in args.materials.split(',')]
    elif args.type == 'generated':
        print("📊 Generando preview de materiales...")
        sem_types = []
        for sem_type_str in args.semiconductor_types:
            if sem_type_str == 'III_V':
                sem_types.append(SemiconductorType.III_V)
            elif sem_type_str == 'II_VI':
                sem_types.append(SemiconductorType.II_VI)
        
        for sem_type in sem_types:
            if sem_type == SemiconductorType.III_V:
                result = MATERIAL_PERMUTATOR.generate_iii_v_combinations()
            elif sem_type == SemiconductorType.II_VI:
                result = MATERIAL_PERMUTATOR.generate_ii_vi_combinations()
            
            materials.extend([sc.formula for sc in result.filtered_combinations[:args.max_materials//len(sem_types)]])
    else:
        # Usar semiconductores comunes
        config = create_common_semiconductors_config()
        materials = [m.formula for m in config.get_enabled_materials()]
    
    # Validar materiales
    valid_materials = []
    invalid_materials = []
    
    for material in materials:
        if material in SEMICONDUCTOR_DB.semiconductors:
            sem = SEMICONDUCTOR_DB.semiconductors[material]
            valid_materials.append(material)
            print(f"✅ {material} ({sem.semiconductor_type.value})")
        else:
            invalid_materials.append(material)
            print(f"❌ {material} (no encontrado)")
    
    print(f"\n📊 Resumen de validación:")
    print(f"   • Válidos: {len(valid_materials)}")
    print(f"   • Inválidos: {len(invalid_materials)}")
    
    if invalid_materials:
        print(f"   • Materiales inválidos: {invalid_materials}")
    
    return len(valid_materials) > 0


def main():
    """Función principal."""
    args = parse_arguments()
    
    # Mostrar banner
    print("=" * 60)
    print("🚀 PIPELINE DE PRECONVERGENCIA MULTIMATERIAL")
    print("=" * 60)
    
    # Solo validación si se solicita
    if args.validate_only:
        success = asyncio.run(validate_materials(args))
        sys.exit(0 if success else 1)
    
    # Ejecutar campaña
    success = asyncio.run(run_campaign(args))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()