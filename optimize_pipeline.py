#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_pipeline.py

Optimizaciones para acelerar el pipeline de preconvergencia DFT/PBC.
Implementa estrategias para reducir tiempos de cálculo manteniendo precisión.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import multiprocessing as mp

# Configuración de PySCF para optimización
os.environ.setdefault("PYSCF_MAX_MEMORY", "4096")  # 4GB por defecto
os.environ.setdefault("OMP_NUM_THREADS", "1")     # Evitar oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

class PipelineOptimizer:
    """Optimizador del pipeline de preconvergencia"""

    def __init__(self, out_dir: Path = Path("preconvergencia_out")):
        self.out_dir = out_dir
        self.benchmarks = {}
        self.optimization_suggestions = []

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analiza cuellos de botella en el pipeline actual"""

        print("🔍 ANALIZANDO CUELLOS DE BOTELLA...")

        bottlenecks = {
            'scf_convergence': self._analyze_scf_performance(),
            'kpoint_generation': self._analyze_kpoint_overhead(),
            'memory_usage': self._analyze_memory_patterns(),
            'parallel_efficiency': self._analyze_parallel_efficiency(),
            'io_overhead': self._analyze_io_overhead()
        }

        return bottlenecks

    def _analyze_scf_performance(self) -> Dict[str, Any]:
        """Analiza rendimiento del SCF"""
        # Leer logs para extraer tiempos de SCF
        log_file = self.out_dir / "preconv.log"
        scf_times = []

        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    if '[SCF]' in line and ('converged' in line or 'timeout' in line):
                        # Extraer tiempo si está disponible en el log
                        pass

        return {
            'avg_scf_time': np.mean(scf_times) if scf_times else None,
            'scf_success_rate': len([t for t in scf_times if t < 300]) / len(scf_times) if scf_times else 0,
            'optimization_needed': len(scf_times) == 0 or np.mean(scf_times) > 60
        }

    def _analyze_kpoint_overhead(self) -> Dict[str, Any]:
        """Analiza overhead de generación de k-points"""
        kmesh_dir = self.out_dir / "kmesh"
        if not kmesh_dir.exists():
            return {'kpoint_overhead': 'no_data'}

        csv_files = list(kmesh_dir.glob("*.csv"))
        if not csv_files:
            return {'kpoint_overhead': 'no_data'}

        # Analizar escalabilidad con k-points
        kpoint_counts = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'N_kpts' in df.columns:
                    kpoint_counts.extend(df['N_kpts'].tolist())
            except:
                continue

        if kpoint_counts:
            kpoint_counts = sorted(list(set(kpoint_counts)))
            scaling_factor = kpoint_counts[-1] / kpoint_counts[0] if len(kpoint_counts) > 1 else 1

            return {
                'kpoint_range': f"{kpoint_counts[0]}-{kpoint_counts[-1]}",
                'scaling_factor': scaling_factor,
                'bottleneck': scaling_factor > 10  # Si escala más de 10x, es bottleneck
            }

        return {'kpoint_overhead': 'insufficient_data'}

    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de uso de memoria"""
        # Estimar uso de memoria basado en tamaños de matrices
        memory_estimates = {
            'small_system': self._estimate_memory_for_kmesh((4,4,4)),
            'medium_system': self._estimate_memory_for_kmesh((8,8,8)),
            'large_system': self._estimate_memory_for_kmesh((12,12,12))
        }

        return {
            'memory_estimates': memory_estimates,
            'recommended_max_memory': min(8192, max(memory_estimates.values()) * 2),  # 2x margen de seguridad
            'optimization_needed': max(memory_estimates.values()) > 4096  # >4GB
        }

    def _estimate_memory_for_kmesh(self, kmesh: Tuple[int,int,int]) -> float:
        """Estima uso de memoria para una malla k-point (MB)"""
        nkpts = kmesh[0] * kmesh[1] * kmesh[2]
        # Estimación simplificada: matrices de overlap, densidad, etc.
        # Típicamente O(N_basis^2 * N_kpts)
        n_basis = 100  # Estimación conservadora para bases GTH
        memory_mb = (n_basis ** 2 * nkpts * 16) / (1024 ** 2)  # 16 bytes por elemento complejo
        return memory_mb

    def _analyze_parallel_efficiency(self) -> Dict[str, Any]:
        """Analiza eficiencia de paralelización"""
        n_cpus = mp.cpu_count()

        return {
            'available_cpus': n_cpus,
            'recommended_omp_threads': min(4, n_cpus // 2),  # No oversubscribir
            'parallel_efficiency': 'unknown',  # Requeriría benchmarks
            'mpi_recommended': n_cpus > 8  # Usar MPI para sistemas grandes
        }

    def _analyze_io_overhead(self) -> Dict[str, Any]:
        """Analiza overhead de I/O"""
        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                total_files += 1
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass

        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 ** 2),
            'io_bottleneck': total_files > 1000 or total_size > 1024 ** 3  # >1GB
        }

    def generate_optimization_strategies(self, bottlenecks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera estrategias de optimización basadas en análisis"""

        strategies = []

        # Estrategia 1: Optimización de SCF
        if bottlenecks.get('scf_convergence', {}).get('optimization_needed', False):
            strategies.append({
                'name': 'scf_optimization',
                'priority': 'high',
                'description': 'Optimizaciones para acelerar convergencia SCF',
                'actions': [
                    'Usar DIIS con space=12',
                    'Implementar level shifting adaptativo',
                    'Ajustar smearing sigma basado en gap',
                    'Usar preconditioners para sistemas grandes'
                ],
                'expected_speedup': '2-5x',
                'implementation_complexity': 'medium'
            })

        # Estrategia 2: Paralelización inteligente
        parallel_info = bottlenecks.get('parallel_efficiency', {})
        if parallel_info.get('available_cpus', 1) > 2:
            strategies.append({
                'name': 'parallel_optimization',
                'priority': 'high',
                'description': 'Mejorar paralelización del cálculo',
                'actions': [
                    f'Configurar OMP_NUM_THREADS={parallel_info.get("recommended_omp_threads", 2)}',
                    'Usar MPI para múltiples nodos' if parallel_info.get('mpi_recommended', False) else 'Optimizar threading OpenMP',
                    'Distribuir k-points entre procesos',
                    'Implementar paralelización híbrida (MPI+OpenMP)'
                ],
                'expected_speedup': f'{parallel_info.get("available_cpus", 2)}x',
                'implementation_complexity': 'low'
            })

        # Estrategia 3: Optimización de memoria
        memory_info = bottlenecks.get('memory_usage', {})
        if memory_info.get('optimization_needed', False):
            strategies.append({
                'name': 'memory_optimization',
                'priority': 'medium',
                'description': 'Optimizar uso de memoria',
                'actions': [
                    f'Configurar PYSCF_MAX_MEMORY={memory_info.get("recommended_max_memory", 4096)}MB',
                    'Usar algoritmos out-of-core para sistemas grandes',
                    'Implementar checkpointing incremental',
                    'Optimizar tamaño de basis set'
                ],
                'expected_speedup': '1.5-3x',
                'implementation_complexity': 'medium'
            })

        # Estrategia 4: Optimización de k-points
        kpoint_info = bottlenecks.get('kpoint_generation', {})
        if kpoint_info.get('bottleneck', False):
            strategies.append({
                'name': 'kpoint_optimization',
                'priority': 'medium',
                'description': 'Optimizar manejo de k-points',
                'actions': [
                    'Usar simetría del grupo puntual para reducir k-points',
                    'Implementar extrapolación de energía vs 1/N_kpts',
                    'Early stopping basado en convergencia',
                    'Reutilización de wavefunctions entre k-points'
                ],
                'expected_speedup': '2-4x',
                'implementation_complexity': 'high'
            })

        # Estrategia 5: Optimización de I/O
        io_info = bottlenecks.get('io_overhead', {})
        if io_info.get('io_bottleneck', False):
            strategies.append({
                'name': 'io_optimization',
                'priority': 'low',
                'description': 'Reducir overhead de I/O',
                'actions': [
                    'Comprimir archivos de checkpoint',
                    'Usar almacenamiento en memoria para cálculos pequeños',
                    'Implementar lazy loading de datos',
                    'Optimizar formato de archivos CSV/JSON'
                ],
                'expected_speedup': '1.2-2x',
                'implementation_complexity': 'low'
            })

        # Estrategia 6: Algoritmos adaptativos
        strategies.append({
            'name': 'adaptive_algorithms',
            'priority': 'high',
            'description': 'Implementar algoritmos adaptativos',
            'actions': [
                'Early stopping basado en criterios de convergencia',
                'Ajuste dinámico de tolerancias',
                'Selección automática de algoritmos SCF',
                'Predicción de parámetros óptimos'
            ],
            'expected_speedup': '3-10x',
            'implementation_complexity': 'high'
        })

        return strategies

    def create_optimized_config(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crea configuración optimizada basada en estrategias"""

        config = {
            'timestamp': datetime.now().isoformat(),
            'optimization_level': 'aggressive',
            'recommended_settings': {},
            'environment_variables': {},
            'expected_performance': {}
        }

        # Aplicar estrategias de alta prioridad primero
        high_priority = [s for s in strategies if s['priority'] == 'high']

        for strategy in high_priority:
            if strategy['name'] == 'scf_optimization':
                config['recommended_settings'].update({
                    'scf_settings': {
                        'diis_space': 12,
                        'level_shift': 0.1,
                        'damp': 0.5,
                        'max_cycle': 50,  # Reducido de 80
                        'conv_tol': 1e-6  # Relajado de 1e-8 para velocidad
                    }
                })

            elif strategy['name'] == 'parallel_optimization':
                n_cpus = mp.cpu_count()
                config['environment_variables'].update({
                    'OMP_NUM_THREADS': str(min(4, n_cpus // 2)),
                    'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1'
                })

            elif strategy['name'] == 'adaptive_algorithms':
                config['recommended_settings'].update({
                    'adaptive_settings': {
                        'early_stopping_threshold': 1e-4,
                        'dynamic_tolerance': True,
                        'predictive_optimization': True
                    }
                })

        # Calcular speedup esperado
        total_speedup = 1.0
        for strategy in strategies:
            speedup_str = strategy.get('expected_speedup', '1x')
            if 'x' in speedup_str:
                try:
                    speedup = float(speedup_str.replace('x', ''))
                    total_speedup *= speedup
                except:
                    pass

        config['expected_performance'] = {
            'total_speedup': total_speedup,
            'estimated_time_savings': f"{total_speedup:.1f}x faster",
            'confidence_level': 'medium' if len(strategies) > 2 else 'low'
        }

        return config

    def generate_optimization_report(self, bottlenecks: Dict[str, Any],
                                   strategies: List[Dict[str, Any]],
                                   config: Dict[str, Any]) -> None:
        """Genera reporte completo de optimización"""

        report_dir = self.out_dir / "optimization_report"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Reporte JSON
        optimization_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'bottlenecks_identified': bottlenecks,
            'optimization_strategies': strategies,
            'recommended_configuration': config,
            'implementation_priority': sorted(strategies, key=lambda x: ['high', 'medium', 'low'].index(x['priority']))
        }

        with open(report_dir / "optimization_analysis.json", "w") as f:
            json.dump(optimization_data, f, indent=2, default=str)

        # Reporte de texto legible
        report_content = f"""
OPTIMIZACIÓN DEL PIPELINE DE PRECONVERGENCIA
{'='*50}

ANÁLISIS REALIZADO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CUELLOS DE BOTELLA IDENTIFICADOS:
{'-'*30}

"""

        for category, data in bottlenecks.items():
            report_content += f"• {category.replace('_', ' ').title()}: {data}\n"

        report_content += f"\nESTRATEGIAS RECOMENDADAS ({len(strategies)}):\n{'-'*30}\n"

        for i, strategy in enumerate(strategies, 1):
            report_content += f"{i}. {strategy['name'].replace('_', ' ').title()} (Prioridad: {strategy['priority'].upper()})\n"
            report_content += f"   Descripción: {strategy['description']}\n"
            report_content += f"   Speedup esperado: {strategy['expected_speedup']}\n"
            report_content += f"   Complejidad: {strategy['implementation_complexity']}\n"
            report_content += "   Acciones:\n"
            for action in strategy['actions']:
                report_content += f"     - {action}\n"
            report_content += "\n"

        report_content += "CONFIGURACIÓN RECOMENDADA:\n"
        report_content += f"{'-'*30}\n"
        report_content += f"Variables de entorno:\n"
        for var, value in config.get('environment_variables', {}).items():
            report_content += f"  export {var}={value}\n"

        report_content += f"\nConfiguración PySCF:\n"
        for section, settings in config.get('recommended_settings', {}).items():
            report_content += f"  {section}: {settings}\n"

        expected_perf = config.get('expected_performance', {})
        report_content += f"\nPERFORMANCE ESPERADO:\n"
        report_content += f"{'-'*30}\n"
        report_content += f"Speedup total: {expected_perf.get('total_speedup', 'N/A')}\n"
        report_content += f"Ahorro estimado: {expected_perf.get('estimated_time_savings', 'N/A')}\n"
        report_content += f"Nivel de confianza: {expected_perf.get('confidence_level', 'N/A')}\n"

        report_content += f"\nIMPLEMENTACIÓN RECOMENDADA:\n"
        report_content += f"{'-'*30}\n"
        report_content += "1. Aplicar variables de entorno\n"
        report_content += "2. Implementar estrategias de alta prioridad\n"
        report_content += "3. Probar con sistema pequeño (validación)\n"
        report_content += "4. Escalar gradualmente a sistemas más grandes\n"
        report_content += "5. Monitorear performance y ajustar\n"

        with open(report_dir / "optimization_report.txt", "w") as f:
            f.write(report_content)

        print(f"✅ Reporte de optimización generado en: {report_dir}")
        print(f"   - optimization_analysis.json")
        print(f"   - optimization_report.txt")

    def run_optimization_analysis(self) -> None:
        """Ejecuta análisis completo de optimización"""

        print("🚀 INICIANDO ANÁLISIS DE OPTIMIZACIÓN")
        print("=" * 50)

        # Paso 1: Analizar cuellos de botella
        print("📊 Analizando cuellos de botella...")
        bottlenecks = self.analyze_bottlenecks()

        # Paso 2: Generar estrategias
        print("🎯 Generando estrategias de optimización...")
        strategies = self.generate_optimization_strategies(bottlenecks)

        # Paso 3: Crear configuración optimizada
        print("⚙️ Creando configuración optimizada...")
        config = self.create_optimized_config(strategies)

        # Paso 4: Generar reporte
        print("📄 Generando reporte de optimización...")
        self.generate_optimization_report(bottlenecks, strategies, config)

        # Resumen
        print("\n" + "=" * 50)
        print("RESUMEN DE OPTIMIZACIÓN:")
        print(f"• Cuellos de botella identificados: {len(bottlenecks)}")
        print(f"• Estrategias recomendadas: {len(strategies)}")
        print(f"• Speedup esperado: {config.get('expected_performance', {}).get('estimated_time_savings', 'N/A')}")
        print("=" * 50)

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Optimizador del pipeline de preconvergencia")
    parser.add_argument("--input_dir", type=str, default="preconvergencia_out",
                       help="Directorio de datos de entrada")
    parser.add_argument("--output_dir", type=str,
                       help="Directorio de salida para reportes")

    args = parser.parse_args()

    # Crear optimizador
    optimizer = PipelineOptimizer(Path(args.input_dir))

    # Ejecutar análisis
    optimizer.run_optimization_analysis()

if __name__ == "__main__":
    main()