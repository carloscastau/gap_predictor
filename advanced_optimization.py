#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script avanzado para optimización global de geometrías en GaAs
Implementa técnicas de búsqueda global con múltiples estrategias
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preconvergencia_GaAs import (
    advanced_lattice_optimization, validate_gaas_parameters,
    build_gaas_cell, run_scf, extract_gap_from_kmf,
    make_kpts, fmt_tuple
)

def run_comprehensive_optimization(config_file=None, **kwargs):
    """
    Ejecuta optimización comprehensiva con múltiples estrategias.

    Args:
        config_file: Archivo JSON con configuración
        **kwargs: Parámetros adicionales para override
    """

    # Configuración por defecto
    config = {
        "a0": 5.653,
        "da": 0.03,
        "npoints_side": 5,
        "x_ga": 0.25,
        "basis": "gth-tzv2p",
        "pseudo": "gth-pbe",
        "xc": "PBE",
        "sigma_ha": 0.01,
        "cutoff_list": [100, 150, 200],
        "kmesh": (8, 8, 8),
        "n_random_restarts": 5,
        "max_refine_iterations": 3,
        "timeout_s": 900,
        "output_dir": "advanced_optimization_results",
        "enable_adaptive_sampling": True,
        "convergence_threshold": 1e-5,
        "min_separation": 0.02  # Separación mínima entre puntos (Å)
    }

    # Cargar configuración desde archivo si existe
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)

    # Override con argumentos de línea de comando
    config.update(kwargs)

    print("🚀 OPTIMIZACIÓN AVANZADA DE GEOMETRÍAS - GaAs")
    print("=" * 60)
    print(f"Configuración: {json.dumps(config, indent=2)}")

    # Crear directorio de salida
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validar parámetros iniciales
    validation = validate_gaas_parameters(config["a0"], config["x_ga"])
    print("
📋 VALIDACIÓN INICIAL:"    print(f"   Parámetro de red: {config['a0']:.4f} Å")
    print(f"   Posición Ga: {config['x_ga']}")
    print(f"   Distancia Ga-As: {validation['distance_Ga_As']:.4f} Å")

    if not validation['valid']:
        print("❌ Problemas detectados en parámetros iniciales:")
        for issue in validation['issues']:
            print(f"   - {issue}")
        return False

    # Configurar logging
    import logging
    log = logging.getLogger("advanced_opt")
    log.setLevel(logging.INFO)

    # Remover handlers existentes
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Crear nuevo handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"optimization_{timestamp}.log"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)

    try:
        # Ejecutar optimización avanzada
        log.info("Iniciando optimización avanzada...")

        df_result, fit_info = advanced_lattice_optimization(
            a0=config["a0"],
            da=config["da"],
            npoints_side=config["npoints_side"],
            x_ga=config["x_ga"],
            basis=config["basis"],
            pseudo=config["pseudo"],
            sigma_ha=config["sigma_ha"],
            xc=config["xc"],
            ke_cutoff_Ry=config["cutoff_list"][-1],  # Usar el cutoff más alto
            kmesh=config["kmesh"],
            out_dir=out_dir,
            log=log,
            nprocs=1,
            timeout_s=config["timeout_s"],
            n_random_restarts=config["n_random_restarts"],
            enable_multi_start=True
        )

        if fit_info is None:
            log.error("No se pudo completar la optimización")
            return False

        # Análisis adicional de resultados
        log.info("📊 ANÁLISIS ADICIONAL DE RESULTADOS")

        # Estadísticas detalladas
        valid_points = df_result.dropna(subset=["E_tot_Ha"])
        if not valid_points.empty:
            energies = valid_points["E_tot_Ha"]
            a_values = valid_points["a_Ang"]

            energy_range = energies.max() - energies.min()
            a_range = a_values.max() - a_values.min()

            log.info(f"Rango de energías: {energy_range*1000:.3f} meV")
            log.info(f"Rango de parámetros: {a_range:.4f} Å")
            log.info(f"Eficiencia del muestreo: {len(valid_points)} puntos válidos")

            # Análisis de calidad del ajuste
            r2 = fit_info.get('R2', np.nan)
            sigma_a = fit_info.get('sigma_aopt', np.nan)

            log.info(f"Calidad del ajuste: R² = {r2:.6f}")
            log.info(f"Precisión del parámetro: σ_a = {sigma_a:.6f} Å")

            # Validación física
            a_opt = fit_info['a_opt']
            if 5.5 <= a_opt <= 5.8:
                log.info(f"✅ Parámetro óptimo en rango físico: {a_opt:.4f} Å")
            else:
                log.warning(f"⚠️ Parámetro óptimo fuera de rango típico: {a_opt:.4f} Å")

        # Crear reporte final
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_method': 'comprehensive_multi_start',
            'configuration': config,
            'results': {
                'a_optimal': fit_info['a_opt'],
                'E_minimum': fit_info['E_min'],
                'fit_R2': fit_info['R2'],
                'fit_sigma_a': fit_info['sigma_aopt'],
                'n_total_evaluations': len(df_result),
                'n_successful_evaluations': len(valid_points),
                'energy_convergence_meV': energy_range * 1000 if 'energy_range' in locals() else None,
                'parameter_range_A': a_range if 'a_range' in locals() else None
            },
            'validation': validation
        }

        # Guardar reporte
        report_file = out_dir / f"final_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)

        log.info(f"✅ Optimización completada exitosamente")
        log.info(f"📁 Resultados guardados en: {out_dir}")
        log.info(f"📄 Reporte final: {report_file}")

        return True

    except Exception as e:
        log.error(f"Error durante la optimización: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Optimización avanzada de geometrías para GaAs")

    # Parámetros básicos
    parser.add_argument("--a0", type=float, default=5.653, help="Parámetro de red inicial (Å)")
    parser.add_argument("--da", type=float, default=0.03, help="Paso de escaneo (Å)")
    parser.add_argument("--npoints_side", type=int, default=5, help="Puntos por lado")
    parser.add_argument("--x_ga", type=float, default=0.25, help="Posición Ga (x,x,x)")

    # Configuración de DFT
    parser.add_argument("--basis", type=str, default="gth-tzv2p", help="Base GTH")
    parser.add_argument("--pseudo", type=str, default="gth-pbe", help="Pseudopotencial")
    parser.add_argument("--xc", type=str, default="PBE", help="Funcional de intercambio-correlación")
    parser.add_argument("--sigma_ha", type=float, default=0.01, help="Smearing (Ha)")

    # Parámetros de convergencia
    parser.add_argument("--cutoff_list", type=str, default="100,150,200", help="Lista de cutoffs (Ry)")
    parser.add_argument("--kmesh", type=str, default="8x8x8", help="Malla de k-points")

    # Estrategias de optimización
    parser.add_argument("--n_random_restarts", type=int, default=5, help="Número de reinicios aleatorios")
    parser.add_argument("--max_refine_iterations", type=int, default=3, help="Máximas iteraciones de refinamiento")
    parser.add_argument("--timeout_s", type=int, default=900, help="Timeout por punto (s)")

    # Configuración de salida
    parser.add_argument("--output_dir", type=str, default="advanced_optimization_results", help="Directorio de salida")
    parser.add_argument("--config_file", type=str, help="Archivo JSON con configuración adicional")

    # Opciones avanzadas
    parser.add_argument("--enable_adaptive_sampling", action="store_true", default=True, help="Habilitar muestreo adaptativo")
    parser.add_argument("--convergence_threshold", type=float, default=1e-5, help="Umbral de convergencia (Ha)")
    parser.add_argument("--min_separation", type=float, default=0.02, help="Separación mínima entre puntos (Å)")

    args = parser.parse_args()

    # Convertir argumentos de cadena a tipos apropiados
    cutoff_list = [float(x.strip()) for x in args.cutoff_list.split(",") if x.strip()]
    kmesh = tuple(int(x) for x in args.kmesh.split("x"))

    # Ejecutar optimización
    success = run_comprehensive_optimization(
        config_file=args.config_file,
        a0=args.a0,
        da=args.da,
        npoints_side=args.npoints_side,
        x_ga=args.x_ga,
        basis=args.basis,
        pseudo=args.pseudo,
        xc=args.xc,
        sigma_ha=args.sigma_ha,
        cutoff_list=cutoff_list,
        kmesh=kmesh,
        n_random_restarts=args.n_random_restarts,
        max_refine_iterations=args.max_refine_iterations,
        timeout_s=args.timeout_s,
        output_dir=args.output_dir,
        enable_adaptive_sampling=args.enable_adaptive_sampling,
        convergence_threshold=args.convergence_threshold,
        min_separation=args.min_separation
    )

    exit_code = 0 if success else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()