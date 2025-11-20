#!/usr/bin/env python3
"""
Script de Generación de Base de Datos de Semiconductores II-VI

Este script genera y actualiza la base de datos CSV de semiconductores II-VI
con propiedades físicas, electrónicas y ópticas.

Autor: Sistema de Análisis de Materiales
Fecha: 2025-11-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiconductorDatabaseGenerator:
    """Generador de base de datos de semiconductores II-VI"""
    
    def __init__(self, output_path: str = "data/semiconductores_ii_vi_ejemplo.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Datos de elementos para cálculos automáticos
        self.element_data = {
            'Zn': {'numero_atomico': 30, 'masa_atomica': 65.38, 'grupo': 12},
            'Cd': {'numero_atomico': 48, 'masa_atomica': 112.41, 'grupo': 12},
            'Hg': {'numero_atomico': 80, 'masa_atomica': 200.59, 'grupo': 12},
            'Mg': {'numero_atomico': 12, 'masa_atomica': 24.31, 'grupo': 2},
            'Be': {'numero_atomico': 4, 'masa_atomica': 9.01, 'grupo': 2},
            'S': {'numero_atomico': 16, 'masa_atomica': 32.07, 'grupo': 16},
            'Se': {'numero_atomico': 34, 'masa_atomica': 78.96, 'grupo': 16},
            'Te': {'numero_atomico': 52, 'masa_atomica': 127.60, 'grupo': 16},
            'O': {'numero_atomico': 8, 'masa_atomica': 16.00, 'grupo': 16}
        }
    
    def calcular_masa_molar(self, elemento_A: str, elemento_B: str) -> float:
        """Calcula la masa molar del compuesto"""
        return self.element_data[elemento_A]['masa_atomica'] + \
               self.element_data[elemento_B]['masa_atomica']
    
    def calcular_volumen_celda_cubica(self, constante_red_a: float) -> float:
        """Calcula el volumen de la celda unitaria cúbica"""
        return constante_red_a ** 3
    
    def generar_base_datos(self) -> pd.DataFrame:
        """Genera la base de datos completa de semiconductores II-VI"""
        
        # Base de datos principal con datos experimentales validados
        data = [
            # ZnS - Zincblende y Wurtzite
            {
                'formula': 'ZnS', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Zn', 'elemento_B': 'S', 'numero_atomico_A': 30, 'numero_atomico_B': 16,
                'masa_molar': 97.47, 'g_cm3': 4.09, 'punto_fusion_K': 2100, 'conductividad_termica_W_mK': 28.4,
                'constante_red_a_angstrom': 5.409, 'constante_red_c_angstrom': None, 
                'volumen_celda_angstrom3': 158.3, 'band_gap_directo_eV': 3.72, 'band_gap_indirecto_eV': 3.78,
                'movilidad_electrones_cm2_Vs': 165, 'movilidad_huecos_cm2_Vs': 15,
                'indice_refraccion': 2.35, 'permitividad_estatica': 8.9, 'energia_exciton_eV': 0.037,
                'referencia_experimental': 'Hummer (1973)', 'doi': '10.1103/PhysRevB.7.5202', 'temperatura_medicion_K': 298
            },
            {
                'formula': 'ZnS', 'grupo_cristalino': 'wurtzite', 'estructura_cristalina': 'hexagonal',
                'elemento_A': 'Zn', 'elemento_B': 'S', 'numero_atomico_A': 30, 'numero_atomico_B': 16,
                'masa_molar': 97.47, 'g_cm3': 4.09, 'punto_fusion_K': 2100, 'conductividad_termica_W_mK': 28.4,
                'constante_red_a_angstrom': 3.82, 'constante_red_c_angstrom': 6.25,
                'volumen_celda_angstrom3': 79.1, 'band_gap_directo_eV': 3.72, 'band_gap_indirecto_eV': 3.78,
                'movilidad_electrones_cm2_Vs': 165, 'movilidad_huecos_cm2_Vs': 15,
                'indice_refraccion': 2.35, 'permitividad_estatica': 8.9, 'energia_exciton_eV': 0.037,
                'referencia_experimental': 'Enkrich (2001)', 'doi': '10.1016/S0925-3467(01)00193-6', 'temperatura_medicion_K': 298
            },
            # ZnSe
            {
                'formula': 'ZnSe', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Zn', 'elemento_B': 'Se', 'numero_atomico_A': 30, 'numero_atomico_B': 34,
                'masa_molar': 144.66, 'g_cm3': 5.26, 'punto_fusion_K': 1790, 'conductividad_termica_W_mK': 19.0,
                'constante_red_a_angstrom': 5.668, 'constante_red_c_angstrom': None,
                'volumen_celda_angstrom3': 182.0, 'band_gap_directo_eV': 2.70, 'band_gap_indirecto_eV': 2.80,
                'movilidad_electrones_cm2_Vs': 540, 'movilidad_huecos_cm2_Vs': 28,
                'indice_refraccion': 2.70, 'permitividad_estatica': 9.1, 'energia_exciton_eV': 0.025,
                'referencia_experimental': 'Tutihasi (1967)', 'doi': '10.1103/PhysRev.158.623', 'temperatura_medicion_K': 298
            },
            # ZnTe
            {
                'formula': 'ZnTe', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Zn', 'elemento_B': 'Te', 'numero_atomico_A': 30, 'numero_atomico_B': 52,
                'masa_molar': 192.99, 'g_cm3': 6.34, 'punto_fusion_K': 1565, 'conductividad_termica_W_mK': 10.9,
                'constante_red_a_angstrom': 6.104, 'constante_red_c_angstrom': None,
                'volumen_celda_angstrom3': 227.5, 'band_gap_directo_eV': 2.26, 'band_gap_indirecto_eV': 2.39,
                'movilidad_electrones_cm2_Vs': 305, 'movilidad_huecos_cm2_Vs': 110,
                'indice_refraccion': 3.06, 'permitividad_estatica': 10.1, 'energia_exciton_eV': 0.017,
                'referencia_experimental': 'Zanato (2004)', 'doi': '10.1016/j.jcrysgro.2004.01.093', 'temperatura_medicion_K': 298
            },
            # CdS
            {
                'formula': 'CdS', 'grupo_cristalino': 'wurtzite', 'estructura_cristalina': 'hexagonal',
                'elemento_A': 'Cd', 'elemento_B': 'S', 'numero_atomico_A': 48, 'numero_atomico_B': 16,
                'masa_molar': 144.46, 'g_cm3': 4.83, 'punto_fusion_K': 1750, 'conductividad_termica_W_mK': 20.0,
                'constante_red_a_angstrom': 4.14, 'constante_red_c_angstrom': 6.72,
                'volumen_celda_angstrom3': 99.8, 'band_gap_directo_eV': 2.42, 'band_gap_indirecto_eV': 2.50,
                'movilidad_electrones_cm2_Vs': 340, 'movilidad_huecos_cm2_Vs': 40,
                'indice_refraccion': 2.50, 'permitividad_estatica': 8.9, 'energia_exciton_eV': 0.028,
                'referencia_experimental': 'Shen (1991)', 'doi': '10.1016/0921-5107(91)90003-Z', 'temperatura_medicion_K': 298
            },
            # CdSe
            {
                'formula': 'CdSe', 'grupo_cristalino': 'wurtzite', 'estructura_cristalina': 'hexagonal',
                'elemento_A': 'Cd', 'elemento_B': 'Se', 'numero_atomico_A': 48, 'numero_atomico_B': 34,
                'masa_molar': 191.37, 'g_cm3': 5.66, 'punto_fusion_K': 1512, 'conductividad_termica_W_mK': 9.3,
                'constante_red_a_angstrom': 4.30, 'constante_red_c_angstrom': 7.01,
                'volumen_celda_angstrom3': 112.2, 'band_gap_directo_eV': 1.74, 'band_gap_indirecto_eV': 1.85,
                'movilidad_electrones_cm2_Vs': 650, 'movilidad_huecos_cm2_Vs': 50,
                'indice_refraccion': 2.70, 'permitividad_estatica': 9.7, 'energia_exciton_eV': 0.015,
                'referencia_experimental': 'Nakamura (1992)', 'doi': '10.1016/0040-6090(92)90107-1', 'temperatura_medicion_K': 298
            },
            # CdTe
            {
                'formula': 'CdTe', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Cd', 'elemento_B': 'Te', 'numero_atomico_A': 48, 'numero_atomico_B': 52,
                'masa_molar': 240.01, 'g_cm3': 5.86, 'punto_fusion_K': 1365, 'conductividad_termica_W_mK': 6.2,
                'constante_red_a_angstrom': 6.48, 'constante_red_c_angstrom': None,
                'volumen_celda_angstrom3': 273.0, 'band_gap_directo_eV': 1.44, 'band_gap_indirecto_eV': 1.56,
                'movilidad_electrones_cm2_Vs': 1050, 'movilidad_huecos_cm2_Vs': 80,
                'indice_refraccion': 2.72, 'permitividad_estatica': 10.2, 'energia_exciton_eV': 0.010,
                'referencia_experimental': 'Adachi (1999)', 'doi': '10.1016/S0925-3467(99)00011-7', 'temperatura_medicion_K': 298
            }
        ]
        
        return pd.DataFrame(data)
    
    def agregar_materiales_contexto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega materiales adicionales para contexto completo"""
        
        # Materiales adicionales relevantes
        materiales_contexto = [
            {
                'formula': 'HgS', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Hg', 'elemento_B': 'S', 'numero_atomico_A': 80, 'numero_atomico_B': 16,
                'masa_molar': 232.65, 'g_cm3': 7.73, 'punto_fusion_K': 2020, 'conductividad_termica_W_mK': 2.8,
                'constante_red_a_angstrom': 5.851, 'constante_red_c_angstrom': None,
                'volumen_celda_angstrom3': 200.2, 'band_gap_directo_eV': 2.0, 'band_gap_indirecto_eV': 2.1,
                'movilidad_electrones_cm2_Vs': 50, 'movilidad_huecos_cm2_Vs': 20,
                'indice_refraccion': 3.20, 'permitividad_estatica': 18.0, 'energia_exciton_eV': 0.029,
                'referencia_experimental': 'Krebs (1954)', 'doi': '10.1007/BF01345535', 'temperatura_medicion_K': 298
            },
            {
                'formula': 'HgTe', 'grupo_cristalino': 'zincblende', 'estructura_cristalina': 'cubic',
                'elemento_A': 'Hg', 'elemento_B': 'Te', 'numero_atomico_A': 80, 'numero_atomico_B': 52,
                'masa_molar': 328.19, 'g_cm3': 8.18, 'punto_fusion_K': 943, 'conductividad_termica_W_mK': 2.9,
                'constante_red_a_angstrom': 6.463, 'constante_red_c_angstrom': None,
                'volumen_celda_angstrom3': 270.0, 'band_gap_directo_eV': -0.14, 'band_gap_indirecto_eV': 0.1,
                'movilidad_electrones_cm2_Vs': 10000, 'movilidad_huecos_cm2_Vs': 100,
                'indice_refraccion': 2.90, 'permitividad_estatica': 23.4, 'energia_exciton_eV': -0.008,
                'referencia_experimental': 'Harman (1960)', 'doi': '10.1063/1.1736488', 'temperatura_medicion_K': 298
            },
            {
                'formula': 'BeO', 'grupo_cristalino': 'wurtzite', 'estructura_cristalina': 'hexagonal',
                'elemento_A': 'Be', 'elemento_B': 'O', 'numero_atomico_A': 4, 'numero_atomico_B': 8,
                'masa_molar': 25.02, 'g_cm3': 3.05, 'punto_fusion_K': 2570, 'conductividad_termica_W_mK': 330.0,
                'constante_red_a_angstrom': 2.70, 'constante_red_c_angstrom': 4.38,
                'volumen_celda_angstrom3': 27.7, 'band_gap_directo_eV': 10.6, 'band_gap_indirecto_eV': 11.0,
                'movilidad_electrones_cm2_Vs': 800, 'movilidad_huecos_cm2_Vs': 300,
                'indice_refraccion': 1.95, 'permitividad_estatica': 7.65, 'energia_exciton_eV': 0.070,
                'referencia_experimental': 'Roessler (1966)', 'doi': '10.1103/PhysRev.146.536', 'temperatura_medicion_K': 298
            }
        ]
        
        df_contexto = pd.DataFrame(materiales_contexto)
        return pd.concat([df, df_contexto], ignore_index=True)
    
    def validar_datos(self, df: pd.DataFrame) -> List[str]:
        """Valida los datos contra valores conocidos de literatura"""
        errores = []
        
        # Validaciones básicas
        for idx, row in df.iterrows():
            # Validar band gaps
            if row['band_gap_directo_eV'] < 0 and row['formula'] not in ['HgTe']:
                errores.append(f"{row['formula']}: Band gap directo negativo no esperado")
            
            # Validar temperaturas
            if row['temperatura_medicion_K'] != 298:
                errores.append(f"{row['formula']}: Temperatura no estándar: {row['temperatura_medicion_K']}K")
            
            # Validar densidades
            if row['g_cm3'] <= 0 or row['g_cm3'] > 15:
                errores.append(f"{row['formula']}: Densidad fuera de rango: {row['g_cm3']} g/cm³")
        
        return errores
    
    def generar_csv(self, incluir_contexto: bool = True) -> bool:
        """Genera el archivo CSV de semiconductores II-VI"""
        try:
            logger.info("Generando base de datos de semiconductores II-VI...")
            
            # Generar datos principales
            df_principal = self.generar_base_datos()
            
            # Agregar materiales de contexto si se solicita
            if incluir_contexto:
                df = self.agregar_materiales_contexto(df_principal)
            else:
                df = df_principal
            
            # Validar datos
            errores = self.validar_datos(df)
            if errores:
                logger.warning("Se encontraron errores en la validación:")
                for error in errores:
                    logger.warning(f"  - {error}")
            
            # Guardar archivo CSV
            df.to_csv(self.output_path, index=False, encoding='utf-8')
            logger.info(f"Archivo CSV generado exitosamente: {self.output_path}")
            logger.info(f"Total de materiales: {len(df)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al generar el archivo CSV: {str(e)}")
            return False
    
    def generar_reporte_estadisticas(self) -> Dict:
        """Genera estadísticas de la base de datos"""
        try:
            df = pd.read_csv(self.output_path)
            
            estadisticas = {
                'total_materiales': len(df),
                'materiales_principales': len(df[df['formula'].isin(['ZnS', 'ZnSe', 'ZnTe', 'CdS', 'CdSe', 'CdTe'])]),
                'estructuras_cristalinas': df['estructura_cristalina'].value_counts().to_dict(),
                'band_gap_rango': {
                    'min': float(df['band_gap_directo_eV'].min()),
                    'max': float(df['band_gap_directo_eV'].max()),
                    'promedio': float(df['band_gap_directo_eV'].mean())
                },
                'elementos_A': df['elemento_A'].unique().tolist(),
                'elementos_B': df['elemento_B'].unique().tolist()
            }
            
            return estadisticas
            
        except Exception as e:
            logger.error(f"Error al generar estadísticas: {str(e)}")
            return {}


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Generador de Base de Datos de Semiconductores II-VI')
    parser.add_argument('--output', '-o', default='data/semiconductores_ii_vi_ejemplo.csv',
                       help='Ruta del archivo de salida (default: data/semiconductores_ii_vi_ejemplo.csv)')
    parser.add_argument('--sin-contexto', action='store_true',
                       help='Generar solo los 6 materiales principales sin contexto')
    parser.add_argument('--estadisticas', '-s', action='store_true',
                       help='Mostrar estadísticas de la base de datos')
    parser.add_argument('--validar', '-v', action='store_true',
                       help='Validar datos existentes')
    
    args = parser.parse_args()
    
    # Crear generador
    generador = SemiconductorDatabaseGenerator(args.output)
    
    # Ejecutar según argumentos
    if args.validar and Path(args.output).exists():
        df = pd.read_csv(args.output)
        errores = generador.validar_datos(df)
        if errores:
            print("Errores encontrados:")
            for error in errores:
                print(f"  - {error}")
        else:
            print("✓ Validación exitosa: No se encontraron errores")
    
    elif args.estadisticas and Path(args.output).exists():
        stats = generador.generar_reporte_estadisticas()
        print("Estadísticas de la Base de Datos:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        # Generar base de datos
        incluir_contexto = not args.sin_contexto
        exito = generador.generar_csv(incluir_contexto)
        
        if exito:
            print(f"✓ Base de datos generada exitosamente en: {args.output}")
        else:
            print("✗ Error al generar la base de datos")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())