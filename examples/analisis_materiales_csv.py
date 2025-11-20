#!/usr/bin/env python3
# examples/analisis_materiales_csv.py
"""
Análisis de Materiales Específicos del CSV de Semiconductores II-VI

Este script demuestra cómo usar los datos experimentales del CSV para:
- Análisis de propiedades específicas
- Comparación de precisión teórica vs experimental
- Identificación de tendencias en semiconductores II-VI
- Filtrado por propiedades objetivo

Ejecutar: python examples/analisis_materiales_csv.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.semiconductor_database import SEMICONDUCTOR_DB
from core.material_permutator import generate_all_ii_vi, PermutationFilter
from analysis.multi_material_analysis import MultiMaterialAnalyzer


def cargar_datos_csv():
    """Carga los datos del CSV de semiconductores II-VI."""
    csv_path = Path(__file__).parent.parent / "data" / "semiconductores_ii_vi_ejemplo.csv"
    
    if not csv_path.exists():
        print(f"❌ Archivo CSV no encontrado: {csv_path}")
        return None
    
    print(f"📁 Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Datos cargados: {len(df)} materiales")
    print(f"📊 Columnas disponibles: {list(df.columns)}")
    
    return df


def analizar_propiedades_experimentales(df):
    """Analiza las propiedades experimentales del CSV."""
    print("\n🧪 ANÁLISIS DE PROPIEDADES EXPERIMENTALES")
    print("=" * 50)
    
    # Estadísticas básicas
    print(f"📈 Estadísticas Generales:")
    print(f"   • Total materiales: {len(df)}")
    print(f"   • Materiales con band gap: {df['bandgap_exp'].notna().sum()}")
    print(f"   • Materiales con lattice: {df['lattice_constant_exp'].notna().sum()}")
    print(f"   • Materiales con bond length: {df['bond_length'].notna().sum()}")
    
    # Análisis de band gaps
    if 'bandgap_exp' in df.columns:
        bandgaps = df['bandgap_exp'].dropna()
        print(f"\n📊 Band Gaps Experimentales:")
        print(f"   • Rango: {bandgaps.min():.2f} - {bandgaps.max():.2f} eV")
        print(f"   • Promedio: {bandgaps.mean():.2f} eV")
        print(f"   • Desviación: {bandgaps.std():.2f} eV")
    
    # Análisis de constantes de red
    if 'lattice_constant_exp' in df.columns:
        lattices = df['lattice_constant_exp'].dropna()
        print(f"\n🔬 Constantes de Red Experimentales:")
        print(f"   • Rango: {lattices.min():.3f} - {lattices.max():.3f} Å")
        print(f"   • Promedio: {lattices.mean():.3f} Å")
        print(f"   • Desviación: {lattices.std():.3f} Å")
    
    # Distribución por estructura cristalina
    if 'crystal_structure' in df.columns:
        structures = df['crystal_structure'].value_counts()
        print(f"\n🏗️ Estructuras Cristalinas:")
        for structure, count in structures.items():
            print(f"   • {structure}: {count} materiales")
    
    return df


def comparar_teorico_vs_experimental(df):
    """Compara valores teóricos vs experimentales."""
    print("\n⚖️ COMPARACIÓN TEÓRICO vs EXPERIMENTAL")
    print("=" * 50)
    
    # Comparar band gaps
    if 'bandgap_exp' in df.columns and 'bandgap_calc' in df.columns:
        print("🔍 Analizando band gaps...")
        
        # Filtrar materiales con ambos valores
        valid_bg = df.dropna(subset=['bandgap_exp', 'bandgap_calc'])
        
        if len(valid_bg) > 0:
            # Calcular errores
            errors_bg = abs(valid_bg['bandgap_calc'] - valid_bg['bandgap_exp'])
            relative_errors_bg = errors_bg / valid_bg['bandgap_exp'] * 100
            
            print(f"   📊 Band Gap:")
            print(f"      • Materiales comparables: {len(valid_bg)}")
            print(f"      • Error promedio: {errors_bg.mean():.3f} eV")
            print(f"      • Error relativo promedio: {relative_errors_bg.mean():.1f}%")
            print(f"      • Error máximo: {errors_bg.max():.3f} eV")
            
            # Mostrar peores y mejores
            valid_bg_copy = valid_bg.copy()
            valid_bg_copy['error'] = errors_bg
            valid_bg_copy['rel_error'] = relative_errors_bg
            
            worst = valid_bg_copy.nlargest(3, 'error')
            best = valid_bg_copy.nsmallest(3, 'error')
            
            print(f"      • Peores predicciones:")
            for _, row in worst.iterrows():
                print(f"        - {row['formula']}: {row['bandgap_exp']:.2f} vs {row['bandgap_calc']:.2f} eV (error: {row['error']:.3f})")
            
            print(f"      • Mejores predicciones:")
            for _, row in best.iterrows():
                print(f"        - {row['formula']}: {row['bandgap_exp']:.2f} vs {row['bandgap_calc']:.2f} eV (error: {row['error']:.3f})")
    
    # Comparar constantes de red
    if 'lattice_constant_exp' in df.columns and 'lattice_constant_calc' in df.columns:
        print("\n🔍 Analizando constantes de red...")
        
        valid_lattice = df.dropna(subset=['lattice_constant_exp', 'lattice_constant_calc'])
        
        if len(valid_lattice) > 0:
            errors_lattice = abs(valid_lattice['lattice_constant_calc'] - valid_lattice['lattice_constant_exp'])
            relative_errors_lattice = errors_lattice / valid_lattice['lattice_constant_exp'] * 100
            
            print(f"   📊 Constante de Red:")
            print(f"      • Materiales comparables: {len(valid_lattice)}")
            print(f"      • Error promedio: {errors_lattice.mean():.4f} Å")
            print(f"      • Error relativo promedio: {relative_errors_lattice.mean():.2f}%")
            print(f"      • Error máximo: {errors_lattice.max():.4f} Å")
            
            # Mostrar distribución de errores
            errors_in_1pct = (relative_errors_lattice <= 1.0).sum()
            errors_in_2pct = (relative_errors_lattice <= 2.0).sum()
            
            print(f"      • Dentro del 1%: {errors_in_1pct}/{len(valid_lattice)} ({errors_in_1pct/len(valid_lattice)*100:.1f}%)")
            print(f"      • Dentro del 2%: {errors_in_2pct}/{len(valid_lattice)} ({errors_in_2pct/len(valid_lattice)*100:.1f}%)")


def filtrar_por_aplicacion(df):
    """Filtra materiales por aplicación específica."""
    print("\n🎯 FILTRADO POR APLICACIÓN")
    print("=" * 35)
    
    # Aplicaciones típicas
    aplicaciones = {
        "LED_azul": {"band_gap_min": 2.5, "band_gap_max": 3.5, "estructura": ["zincblende", "wurtzite"]},
        "LED_verde": {"band_gap_min": 2.0, "band_gap_max": 2.5, "estructura": ["zincblende", "wurtzite"]},
        "Solar_cells": {"band_gap_min": 1.0, "band_gap_max": 2.0, "estructura": ["zincblende", "wurtzite"]},
        "Infrarrojo": {"band_gap_min": 0.3, "band_gap_max": 1.0, "estructura": ["zincblende"]},
        "Wide_bandgap": {"band_gap_min": 3.0, "band_gap_max": 6.0, "estructura": ["wurtzite", "zincblende"]}
    }
    
    for aplicacion, criterios in aplicaciones.items():
        print(f"\n🔬 Aplicación: {aplicacion}")
        
        # Filtrar por band gap
        if 'bandgap_exp' in df.columns:
            candidatos = df[
                (df['bandgap_exp'] >= criterios['band_gap_min']) &
                (df['bandgap_exp'] <= criterios['band_gap_max'])
            ]
        else:
            candidatos = df.copy()
        
        # Filtrar por estructura si se especifica
        if 'crystal_structure' in df.columns and criterios['estructura']:
            candidatos = candidatos[candidatos['crystal_structure'].isin(criterios['estructura'])]
        
        print(f"   • Candidatos encontrados: {len(candidatos)}")
        
        if len(candidatos) > 0:
            print(f"   • Materiales:")
            for _, row in candidatos.iterrows():
                formula = row['formula']
                bg = row.get('bandgap_exp', 'N/A')
                lattice = row.get('lattice_constant_exp', 'N/A')
                structure = row.get('crystal_structure', 'N/A')
                
                bg_str = f"{bg:.2f}" if pd.notna(bg) else "N/A"
                lattice_str = f"{lattice:.3f}" if pd.notna(lattice) else "N/A"
                
                print(f"     - {formula:8s} | BG: {bg_str:>5s} eV | a: {lattice_str:>6s} Å | {structure}")


def generar_graficos_analisis(df, output_dir):
    """Genera gráficos de análisis."""
    print("\n📊 GENERANDO GRÁFICOS DE ANÁLISIS")
    print("=" * 40)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gráfico 1: Distribución de band gaps
    if 'bandgap_exp' in df.columns:
        plt.figure(figsize=(10, 6))
        bandgaps = df['bandgap_exp'].dropna()
        
        plt.hist(bandgaps, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Band Gap Experimental (eV)')
        plt.ylabel('Número de Materiales')
        plt.title('Distribución de Band Gaps Experimentales - Semiconductores II-VI')
        plt.grid(True, alpha=0.3)
        
        # Añadir estadísticas al gráfico
        plt.axvline(bandgaps.mean(), color='red', linestyle='--', 
                   label=f'Promedio: {bandgaps.mean():.2f} eV')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distribucion_bandgaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Gráfico guardado: distribucion_bandgaps.png")
    
    # Gráfico 2: Constantes de red vs Band gaps
    if 'lattice_constant_exp' in df.columns and 'bandgap_exp' in df.columns:
        plt.figure(figsize=(10, 8))
        
        valid_data = df.dropna(subset=['lattice_constant_exp', 'bandgap_exp'])
        
        scatter = plt.scatter(valid_data['lattice_constant_exp'], 
                            valid_data['bandgap_exp'],
                            c=valid_data.index, 
                            cmap='viridis', 
                            s=100, 
                            alpha=0.7)
        
        plt.xlabel('Constante de Red Experimental (Å)')
        plt.ylabel('Band Gap Experimental (eV)')
        plt.title('Relación: Constante de Red vs Band Gap - Semiconductores II-VI')
        
        # Añadir etiquetas para cada punto
        for _, row in valid_data.iterrows():
            plt.annotate(row['formula'], 
                        (row['lattice_constant_exp'], row['bandgap_exp']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, label='Material Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'lattice_vs_bandgap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Gráfico guardado: lattice_vs_bandgap.png")
    
    # Gráfico 3: Comparación teórico vs experimental
    if 'bandgap_exp' in df.columns and 'bandgap_calc' in df.columns:
        plt.figure(figsize=(8, 8))
        
        valid_comparison = df.dropna(subset=['bandgap_exp', 'bandgap_calc'])
        
        if len(valid_comparison) > 0:
            plt.scatter(valid_comparison['bandgap_exp'], 
                       valid_comparison['bandgap_calc'],
                       alpha=0.7, s=100)
            
            # Línea ideal (y=x)
            min_val = min(valid_comparison['bandgap_exp'].min(), 
                         valid_comparison['bandgap_calc'].min())
            max_val = max(valid_comparison['bandgap_exp'].max(), 
                         valid_comparison['bandgap_calc'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                    label='Línea Ideal (y=x)')
            
            plt.xlabel('Band Gap Experimental (eV)')
            plt.ylabel('Band Gap Calculado (eV)')
            plt.title('Comparación: Teórico vs Experimental - Band Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Añadir etiquetas
            for _, row in valid_comparison.iterrows():
                plt.annotate(row['formula'], 
                            (row['bandgap_exp'], row['bandgap_calc']),
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'comparacion_teorico_experimental.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✅ Gráfico guardado: comparacion_teorico_experimental.png")


def identificar_tendencias(df):
    """Identifica tendencias en los datos."""
    print("\n📈 IDENTIFICACIÓN DE TENDENCIAS")
    print("=" * 40)
    
    # Tendencia 1: Correlación lattice vs band gap
    if 'lattice_constant_exp' in df.columns and 'bandgap_exp' in df.columns:
        valid_data = df.dropna(subset=['lattice_constant_exp', 'bandgap_exp'])
        
        if len(valid_data) > 1:
            correlation = valid_data['lattice_constant_exp'].corr(valid_data['bandgap_exp'])
            print(f"🔗 Correlación Lattice vs Band Gap: {correlation:.3f}")
            
            if abs(correlation) > 0.5:
                direction = "negativa" if correlation < 0 else "positiva"
                print(f"   • Correlación {direction} fuerte detectada")
                
                if correlation < 0:
                    print("   • Tendencia: A mayor constante de red → menor band gap")
                    print("   • Esto es típico en semiconductores II-VI")
            else:
                print("   • Correlación débil, sin tendencia clara")
    
    # Tendencia 2: Análisis por grupos
    if 'formula' in df.columns:
        # Extraer cationes y aniones
        df['cation'] = df['formula'].str[0]
        df['anion'] = df['formula'].str[1:]
        
        # Análisis por cation
        if 'bandgap_exp' in df.columns:
            cation_analysis = df.groupby('cation')['bandgap_exp'].agg(['mean', 'std', 'count'])
            print(f"\n🔬 Análisis por Catión (Band Gap):")
            for cation, stats in cation_analysis.iterrows():
                print(f"   • {cation}: {stats['mean']:.2f}±{stats['std']:.2f} eV (n={stats['count']})")
        
        # Análisis por anión
        if 'lattice_constant_exp' in df.columns:
            anion_analysis = df.groupby('anion')['lattice_constant_exp'].agg(['mean', 'std', 'count'])
            print(f"\n🔬 Análisis por Anión (Constante de Red):")
            for anion, stats in anion_analysis.iterrows():
                print(f"   • {anion}: {stats['mean']:.3f}±{stats['std']:.3f} Å (n={stats['count']})")


def main():
    """Función principal del análisis."""
    print("🔬 ANÁLISIS DE MATERIALES DEL CSV DE SEMICONDUCTORES II-VI")
    print("=" * 65)
    
    # 1. Cargar datos
    df = cargar_datos_csv()
    if df is None:
        return
    
    # 2. Mostrar estructura de datos
    print(f"\n📋 Estructura de Datos:")
    print(df.head())
    
    # 3. Análisis de propiedades experimentales
    analizar_propiedades_experimentales(df)
    
    # 4. Comparación teórico vs experimental
    comparar_teorico_vs_experimental(df)
    
    # 5. Filtrado por aplicación
    filtrar_por_aplicacion(df)
    
    # 6. Identificar tendencias
    identificar_tendencias(df)
    
    # 7. Generar gráficos
    output_dir = Path("results/csv_analysis")
    generar_graficos_analisis(df, output_dir)
    
    # 8. Exportar análisis
    print(f"\n💾 EXPORTANDO ANÁLISIS")
    print("=" * 25)
    
    # Crear resumen estadístico
    resumen = {
        'total_materiales': len(df),
        'materiales_con_bandgap': df['bandgap_exp'].notna().sum() if 'bandgap_exp' in df.columns else 0,
        'materiales_con_lattice': df['lattice_constant_exp'].notna().sum() if 'lattice_constant_exp' in df.columns else 0,
        'estructuras_cristalinas': df['crystal_structure'].value_counts().to_dict() if 'crystal_structure' in df.columns else {},
        'promedio_bandgap': df['bandgap_exp'].mean() if 'bandgap_exp' in df.columns else None,
        'promedio_lattice': df['lattice_constant_exp'].mean() if 'lattice_constant_exp' in df.columns else None
    }
    
    # Guardar resumen
    import json
    with open(output_dir / 'analisis_resumen.json', 'w') as f:
        json.dump(resumen, f, indent=2, default=str)
    
    # Exportar datos filtrados
    if 'bandgap_exp' in df.columns:
        # Materiales para LED azul
        led_azul = df[(df['bandgap_exp'] >= 2.5) & (df['bandgap_exp'] <= 3.5)]
        led_azul.to_csv(output_dir / 'materiales_led_azul.csv', index=False)
        print(f"   ✅ Exportados {len(led_azul)} materiales para LED azul")
        
        # Materiales para células solares
        solar = df[(df['bandgap_exp'] >= 1.0) & (df['bandgap_exp'] <= 2.0)]
        solar.to_csv(output_dir / 'materiales_solar.csv', index=False)
        print(f"   ✅ Exportados {len(solar)} materiales para células solares")
    
    print(f"\n🎉 ANÁLISIS COMPLETADO")
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"   • Gráficos: *.png")
    print(f"   • Datos: *.csv")
    print(f"   • Resumen: analisis_resumen.json")


if __name__ == "__main__":
    main()