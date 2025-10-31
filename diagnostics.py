#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnóstico avanzado para cálculos DFT de GaAs
Analiza resultados y detecta problemas comunes en la convergencia
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def analyze_convergence_results(results_dir="preconvergencia_out"):
    """Análisis completo de resultados de convergencia."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Directorio {results_path} no encontrado")
        return False

    print("🔍 ANÁLISIS DE CONVERGENCIA DFT - GaAs")
    print("=" * 50)

    issues = []
    warnings = []

    # 1. Análisis de cutoff
    cutoff_file = results_path / "cutoff" / "cutoff.csv"
    if cutoff_file.exists():
        print("\n📊 ANÁLISIS DE CUTOFF:")
        df_cutoff = pd.read_csv(cutoff_file)
        print(f"   Puntos calculados: {len(df_cutoff)}")
        print(f"   Rango: {df_cutoff['ke_cutoff_Ry'].min():.0f} - {df_cutoff['ke_cutoff_Ry'].max():.0f} Ry")

        if len(df_cutoff) >= 3:
            energies = df_cutoff['E_tot_Ha'].dropna()
            if len(energies) >= 2:
                max_diff = energies.max() - energies.min()
                print(f"   ΔE máximo: {max_diff*1000:.2f} meV")

                if max_diff < 0.001:  # 1 meV
                    print("   ✅ Convergencia de cutoff adecuada")
                else:
                    issues.append(f"Cutoff no convergido: ΔE = {max_diff*1000:.1f} meV > 1 meV")
            else:
                warnings.append("Datos insuficientes para evaluar convergencia de cutoff")
        else:
            warnings.append("Muy pocos puntos de cutoff para evaluación")

    # 2. Análisis de k-mesh
    kmesh_file = results_path / "kmesh" / "kmesh.csv"
    if kmesh_file.exists():
        print("\n🔲 ANÁLISIS DE K-MESH:")
        df_kmesh = pd.read_csv(kmesh_file)
        print(f"   Puntos calculados: {len(df_kmesh)}")
        print(f"   Rango: {df_kmesh['N_kpts'].min()} - {df_kmesh['N_kpts'].max()} k-points")

        if len(df_kmesh) >= 3:
            energies = df_kmesh['E_tot_Ha'].dropna()
            if len(energies) >= 2:
                max_diff = energies.max() - energies.min()
                print(f"   ΔE máximo: {max_diff*1000:.2f} meV")

                if max_diff < 0.001:  # 1 meV
                    print("   ✅ Convergencia de k-mesh adecuada")
                else:
                    issues.append(f"K-mesh no convergido: ΔE = {max_diff*1000:.1f} meV > 1 meV")
            else:
                warnings.append("Datos insuficientes para evaluar convergencia de k-mesh")
        else:
            warnings.append("Muy pocos puntos de k-mesh para evaluación")

    # 3. Análisis de estructura de bandas (si existe)
    bands_file = results_path / "bands" / "gap_summary.csv"
    if bands_file.exists():
        print("\n🎵 ANÁLISIS DE ESTRUCTURA DE BANDAS:")
        df_bands = pd.read_csv(bands_file)

        for _, row in df_bands.iterrows():
            gap = row.get('gap_eV', np.nan)
            gap_type = row.get('direct_indirect', 'unknown')

            print(f"   Base: {row.get('basis', 'unknown')}")
            print(f"   Gap: {gap:.3f} eV ({gap_type})")
            print(f"   a_opt: {row.get('a_opt_Ang', np.nan):.4f} Å")

            # Validaciones físicas
            if pd.isna(gap) or gap < 0:
                issues.append(f"Gap no válido detectado: {gap} eV")
            elif gap < 0.5:
                warnings.append(f"Gap muy pequeño: {gap:.3f} eV (GaAs experimental: 1.42 eV)")
            elif gap > 2.0:
                warnings.append(f"Gap muy grande: {gap:.3f} eV (GaAs experimental: 1.42 eV)")
            else:
                print("   ✅ Gap en rango físico razonable")

    # 4. Análisis de lattice optimization (si existe)
    lattice_file = results_path / "lattice" / "lattice_scan.csv"
    if lattice_file.exists():
        print("\n📏 ANÁLISIS DE OPTIMIZACIÓN DE RED:")
        df_lattice = pd.read_csv(lattice_file)

        if len(df_lattice) >= 5:
            a_vals = df_lattice['a_Ang'].values
            E_vals = df_lattice['E_tot_Ha'].values

            # Buscar mínimo
            min_idx = np.argmin(E_vals)
            a_opt = a_vals[min_idx]
            E_min = E_vals[min_idx]

            print(f"   a_opt encontrado: {a_opt:.4f} Å")
            print(f"   E_min: {E_min:.6f} Ha")

            # Validación física
            if 5.5 <= a_opt <= 5.8:
                print("   ✅ Parámetro de red en rango físico (GaAs: 5.653 Å)")
            else:
                issues.append(f"Parámetro de red fuera de rango: {a_opt:.3f} Å")

            # Calidad del ajuste
            if 'fit_info' in locals():  # Si se guardó información del ajuste
                r2 = fit_info.get('R2', np.nan)
                if pd.isna(r2) or r2 < 0.99:
                    warnings.append(f"Calidad de ajuste cuestionable: R² = {r2}")
                else:
                    print(f"   ✅ Buena calidad de ajuste: R² = {r2:.4f}")
        else:
            warnings.append("Muy pocos puntos para optimización de lattice")

    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DEL DIAGNÓSTICO:")

    if not issues and not warnings:
        print("✅ Todos los parámetros parecen estar correctamente configurados")
        print("✅ No se detectaron problemas obvios en la convergencia")
    else:
        if issues:
            print(f"\n❌ PROBLEMAS DETECTADOS ({len(issues)}):")
            for issue in issues:
                print(f"   • {issue}")

        if warnings:
            print(f"\n⚠️  ADVERTENCIAS ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")

    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    if issues:
        print("   🔧 Corrija los problemas marcados con ❌ antes de proceder")
    if not issues and warnings:
        print("   ⚡ Considere las advertencias para mejorar la calidad")
    if not issues and not warnings:
        print("   🚀 Puede proceder con confianza a cálculos de producción")

    return len(issues) == 0

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "preconvergencia_out"
    success = analyze_convergence_results(results_dir)
    sys.exit(0 if success else 1)