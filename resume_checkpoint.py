#!/usr/bin/env python3
"""
Script para reanudar cálculos desde checkpoints
"""

import json
import pandas as pd
from pathlib import Path
import sys
import os

def list_available_checkpoints(results_dir="preconvergencia_out"):
    """Lista todos los checkpoints disponibles."""
    checkpoint_dir = Path(results_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"❌ No se encontró directorio de checkpoints: {checkpoint_dir}")
        return []

    checkpoints = []
    for cp_file in checkpoint_dir.glob("checkpoint_*.json"):
        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
                checkpoints.append({
                    'file': cp_file,
                    'stage': data.get('stage'),
                    'timestamp': data.get('timestamp'),
                    'status': data.get('status')
                })
        except Exception as e:
            print(f"⚠️ Error leyendo checkpoint {cp_file}: {e}")

    # Ordenar por timestamp
    checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

    return checkpoints

def analyze_checkpoint_status(results_dir="preconvergencia_out"):
    """Analiza el estado actual de los cálculos."""
    print("🔍 ANÁLISIS DE CHECKPOINTS")
    print("=" * 50)

    checkpoints = list_available_checkpoints(results_dir)

    if not checkpoints:
        print("📁 No se encontraron checkpoints")
        return None

    print(f"📊 Encontrados {len(checkpoints)} checkpoints:")
    print()

    # Separar checkpoints normales e incrementales
    normal_checkpoints = [cp for cp in checkpoints if cp.get('type') != 'incremental']
    incremental_checkpoints = [cp for cp in checkpoints if cp.get('type') == 'incremental']

    print(f"📊 Checkpoints normales: {len(normal_checkpoints)}")
    print(f"📊 Checkpoints incrementales: {len(incremental_checkpoints)}")
    print()

    # Mostrar checkpoints normales
    if normal_checkpoints:
        print("🔄 CHECKPOINTS NORMALES:")
        for i, cp in enumerate(normal_checkpoints[:5], 1):
            print(f"{i}. {cp['stage']} - {cp['timestamp']} ({cp['status']})")
        if len(normal_checkpoints) > 5:
            print(f"... y {len(normal_checkpoints) - 5} más")
        print()

    # Mostrar checkpoints incrementales
    if incremental_checkpoints:
        print("⚡ CHECKPOINTS INCREMENTALES:")
        for i, cp in enumerate(incremental_checkpoints[:5], 1):
            print(f"{i}. {cp['stage']} - {cp['timestamp']}")
        if len(incremental_checkpoints) > 5:
            print(f"... y {len(incremental_checkpoints) - 5} más")
        print()

    print("📈 ESTADO ACTUAL:")

    # Verificar qué etapas están completadas
    stages = ['pre_cutoff', 'post_cutoff', 'pre_kmesh', 'post_kmesh',
              'pre_lattice', 'post_lattice', 'pre_bands', 'post_bands',
              'pre_slab', 'post_slab', 'completed']

    latest_stage = None
    for stage in stages:
        stage_checkpoints = [cp for cp in checkpoints if cp['stage'] == stage]
        if stage_checkpoints:
            latest_stage = stage
            incremental_count = len([cp for cp in stage_checkpoints if cp.get('type') == 'incremental'])
            print(f"   ✅ {stage}: {len(stage_checkpoints)} checkpoints ({incremental_count} incrementales)")
        else:
            print(f"   ❌ {stage}: No completado")

    return latest_stage

def get_resume_suggestions(results_dir="preconvergencia_out"):
    """Proporciona sugerencias para reanudar."""
    print("\n💡 SUGERENCIAS PARA REANUDAR:")
    print("-" * 40)

    checkpoints = list_available_checkpoints(results_dir)

    if not checkpoints:
        print("1. No hay checkpoints disponibles")
        print("2. Ejecutar desde el inicio:")
        print("   python preconvergencia_GaAs.py --fast --timeout_s 300")
        print("   # O usar el pipeline incremental optimizado:")
        print("   python incremental_pipeline.py --fast --reuse_previous on")
        return

    # Separar tipos de checkpoints
    normal_checkpoints = [cp for cp in checkpoints if cp.get('type') != 'incremental']
    incremental_checkpoints = [cp for cp in checkpoints if cp.get('type') == 'incremental']

    # Encontrar el último checkpoint exitoso
    completed_checkpoints = [cp for cp in checkpoints if cp['status'] == 'completed']
    if completed_checkpoints:
        latest_complete = max(completed_checkpoints, key=lambda x: x['timestamp'])
        print(f"1. Último checkpoint completo: {latest_complete['stage']}")
        print("   ✅ Puedes continuar desde aquí o reiniciar")

    # Verificar si hay cálculos en progreso
    in_progress = [cp for cp in checkpoints if cp['status'] == 'in_progress']
    if in_progress:
        latest_in_progress = max(in_progress, key=lambda x: x['timestamp'])
        print(f"2. Checkpoint en progreso: {latest_in_progress['stage']}")
        print("   ⚠️ Puede haber cálculos incompletos")

    # Información sobre checkpoints incrementales
    if incremental_checkpoints:
        print(f"3. Checkpoints incrementales disponibles: {len(incremental_checkpoints)}")
        print("   💡 El pipeline incremental puede reutilizar estos datos")

    print("\n4. Comandos sugeridos:")
    print("   # Para desarrollo rápido (con checkpoints incrementales):")
    print("   python preconvergencia_GaAs.py --fast --timeout_s 300 --npoints_side 2")
    print()
    print("   # Para pipeline incremental optimizado (reutiliza resultados previos):")
    print("   python incremental_pipeline.py --fast --reuse_previous on --target_accuracy 1e-4")
    print()
    print("   # Para análisis de resultados:")
    print("   python diagnostics.py")
    print()
    print("   # Para reanudar desde checkpoint específico:")
    print("   python preconvergencia_GaAs.py --resume on --fast")

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "preconvergencia_out"

    print(f"📂 Analizando directorio: {results_dir}")
    print()

    # Listar checkpoints
    checkpoints = list_available_checkpoints(results_dir)

    if not checkpoints:
        print("❌ No se encontraron checkpoints")
        print("💡 Ejecuta el cálculo principal para generar checkpoints")
        return

    # Análisis de estado
    latest_stage = analyze_checkpoint_status(results_dir)

    # Sugerencias
    get_resume_suggestions(results_dir)

    print("\n📋 RESUMEN:")
    print(f"- Checkpoints totales: {len(checkpoints)}")
    print(f"- Última etapa completada: {latest_stage or 'Ninguna'}")

if __name__ == "__main__":
    main()