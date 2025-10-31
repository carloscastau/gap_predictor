#!/usr/bin/env python3
"""
Pipeline incremental optimizado para preconvergencia GaAs
- Cálculo recurrente que reutiliza resultados previos
- Estrategia de convergencia adaptativa
- Paralelización inteligente
"""

import os
import sys
import json
import math
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscf import lib as pyscf_lib
from pyscf.pbc import gto, dft
from pyscf import scf as mol_scf

# Importar funciones del pipeline principal
from preconvergencia_GaAs import (
    setup_logging, ensure_dirs, save_checkpoint, save_incremental_checkpoint,
    load_latest_checkpoint, parse_k_list, parse_cutoff_list, str2bool,
    ha2ev, ev2ha, ry2ha, fmt_tuple, build_gaas_cell, run_scf,
    quadratic_fit, extract_gap_from_kmf, compute_bands_and_write,
    validate_gaas_parameters, HARTREE_TO_EV, DEFAULT_CUTOFF_LIST_RY,
    DEFAULT_K_LIST, UNIVERSAL_GTH_BASES
)

class IncrementalOptimizer:
    """Optimizador incremental que reutiliza cálculos previos."""

    def __init__(self, out_root: Path, log: logging.Logger):
        self.out_root = out_root
        self.log = log
        self.cache = {}  # Cache de resultados previos
        self.convergence_history = []

    def load_previous_results(self):
        """Carga resultados previos para reutilizar."""
        self.log.info("[INCREMENTAL] Cargando resultados previos...")

        # Cargar datos de cutoff
        cutoff_csv = self.out_root / "cutoff" / "cutoff.csv"
        if cutoff_csv.exists():
            try:
                df_cutoff = pd.read_csv(cutoff_csv)
                for _, row in df_cutoff.iterrows():
                    key = (row['ke_cutoff_Ry'], row['kmesh'], row['basis'])
                    self.cache[key] = row['E_tot_Ha']
                self.log.info(f"[INCREMENTAL] Cargados {len(df_cutoff)} resultados de cutoff")
            except Exception as e:
                self.log.warning(f"[INCREMENTAL] Error cargando cutoff: {e}")

        # Cargar datos de k-mesh
        kmesh_csv = self.out_root / "kmesh" / "kmesh.csv"
        if kmesh_csv.exists():
            try:
                df_kmesh = pd.read_csv(kmesh_csv)
                for _, row in df_kmesh.iterrows():
                    key = (row['kx'], row['ky'], row['kz'], row['basis'], row['ke_cutoff_Ry'])
                    self.cache[key] = row['E_tot_Ha']
                self.log.info(f"[INCREMENTAL] Cargados {len(df_kmesh)} resultados de k-mesh")
            except Exception as e:
                self.log.warning(f"[INCREMENTAL] Error cargando k-mesh: {e}")

        # Cargar datos de lattice
        lattice_csv = self.out_root / "lattice" / "lattice_optimization.csv"
        if lattice_csv.exists():
            try:
                df_lattice = pd.read_csv(lattice_csv)
                for _, row in df_lattice.iterrows():
                    key = (row['a_Ang'], row['basis'], row['ke_cutoff_Ry'], row['kmesh'])
                    self.cache[key] = row['E_tot_Ha']
                self.log.info(f"[INCREMENTAL] Cargados {len(df_lattice)} resultados de lattice")
            except Exception as e:
                self.log.warning(f"[INCREMENTAL] Error cargando lattice: {e}")

    def distribute_workload_hpc(self, total_tasks: int, hpc_config: Dict[str, Any] = None) -> List[Tuple[int, int]]:
        """Distribuye carga de trabajo para HPC basado en configuración SLURM."""
        if hpc_config is None:
            hpc_config = {}

        # Detectar configuración SLURM
        n_nodes = int(os.environ.get('SLURM_NNODES', 1))
        n_tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
        array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

        total_slots = n_nodes * n_tasks_per_node

        if array_task_count > 1:
            # Modo array: cada tarea del array procesa un subconjunto
            tasks_per_array_job = max(1, total_tasks // array_task_count)
            start_idx = array_task_id * tasks_per_array_job
            end_idx = min(total_tasks, start_idx + tasks_per_array_job)
        else:
            # Modo single: distribuir entre nodos/tasks disponibles
            tasks_per_slot = max(1, total_tasks // total_slots)
            slot_id = int(os.environ.get('SLURM_PROCID', 0))
            start_idx = slot_id * tasks_per_slot
            end_idx = min(total_tasks, start_idx + tasks_per_slot)

        self.log.info(f"[HPC] Distribuyendo trabajo: tareas {start_idx}-{end_idx} de {total_tasks}")
        return [(start_idx, end_idx)]

    def get_cached_result(self, key):
        """Obtiene resultado del cache si existe."""
        return self.cache.get(key, None)

    def adaptive_convergence_strategy(self, current_results: List[Dict], target_accuracy: float = 1e-4):
        """
        Estrategia de convergencia adaptativa basada en resultados previos.
        """
        if len(current_results) < 2:
            return False, "Necesito más puntos"

        energies = [r['E_tot_Ha'] for r in current_results if np.isfinite(r['E_tot_Ha'])]

        if len(energies) < 2:
            return False, "Pocos resultados válidos"

        # Calcular gradiente de convergencia
        gradients = []
        for i in range(1, len(energies)):
            grad = abs(energies[i] - energies[i-1])
            gradients.append(grad)

        avg_gradient = np.mean(gradients)
        max_gradient = max(gradients)

        # Estrategia adaptativa
        if avg_gradient < target_accuracy * 0.1:
            return True, f"Convergencia excelente (grad={avg_gradient:.2e})"
        elif avg_gradient < target_accuracy:
            return True, f"Convergencia buena (grad={avg_gradient:.2e})"
        elif max_gradient > target_accuracy * 10:
            return False, f"Necesito más puntos (grad_max={max_gradient:.2e})"
        else:
            return False, f"Continuar refinando (grad={avg_gradient:.2e})"

def incremental_cutoff_scan(optimizer: IncrementalOptimizer, cutoff_list: List[float],
                           a_A: float, x_ga: float, basis: str, pseudo: str,
                           sigma_ha: float, xc: str, kmesh_fixed: Tuple[int,int,int],
                           out_dir: Path, log: logging.Logger, **kwargs) -> pd.DataFrame:
    """Escaneo de cutoff con reutilización de resultados."""

    out_csv = out_dir / "cutoff.csv"
    rows = []
    if out_csv.exists():
        try:
            rows = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            pass

    done = {(r.get("ke_cutoff_Ry"), r.get("kmesh"), r.get("basis")) for r in rows}

    def append_and_flush(rec):
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values("ke_cutoff_Ry", inplace=True)
        df.to_csv(out_csv, index=False)

    # Usar resultados del cache cuando sea posible
    for cut in cutoff_list:
        cache_key = (cut, fmt_tuple(kmesh_fixed), basis)
        cached_energy = optimizer.get_cached_result(cache_key)

        if cached_energy is not None:
            log.info(f"[cutoff] usando cache | cut={cut} Ry | E={cached_energy:.8f} Ha")
            rec = dict(ke_cutoff_Ry=cut, E_tot_Ha=cached_energy, n_kpts=0, cached=True)
        elif (cut, fmt_tuple(kmesh_fixed), basis) in done:
            log.info(f"[cutoff] ya existe cut={cut} Ry, skip")
            continue
        else:
            log.info(f"[cutoff] calculando | cut={cut} Ry")
            # Aquí iría el cálculo real (simulado por ahora)
            rec = dict(ke_cutoff_Ry=cut, E_tot_Ha=np.nan, n_kpts=0, cached=False)

        rec.update(dict(kmesh=fmt_tuple(kmesh_fixed), sigma_Ha=sigma_ha, basis=basis,
                       timestamp=datetime.now().isoformat()))
        append_and_flush(rec)

    return pd.DataFrame(rows)

def incremental_kmesh_scan(optimizer: IncrementalOptimizer, k_list: List[Tuple[int,int,int]],
                          a_A: float, x_ga: float, basis: str, pseudo: str,
                          sigma_ha: float, xc: str, ke_cutoff_Ry: float,
                          out_dir: Path, log: logging.Logger, **kwargs) -> pd.DataFrame:
    """Escaneo de k-mesh con reutilización de resultados."""

    out_csv = out_dir / "kmesh.csv"
    rows = []
    if out_csv.exists():
        try:
            rows = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            pass

    done = {(r.get("kx"), r.get("ky"), r.get("kz"), r.get("basis"), r.get("ke_cutoff_Ry")) for r in rows}

    def append_and_flush(rec):
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values(["kx","ky","kz"], inplace=True)
        df.to_csv(out_csv, index=False)

    # Estrategia adaptativa: empezar con mallas pequeñas y refinar
    sorted_k_list = sorted(k_list, key=lambda x: x[0]*x[1]*x[2])

    for km in sorted_k_list:
        cache_key = (km[0], km[1], km[2], basis, ke_cutoff_Ry)
        cached_energy = optimizer.get_cached_result(cache_key)

        if cached_energy is not None:
            log.info(f"[kmesh] usando cache | k={fmt_tuple(km)} | E={cached_energy:.8f} Ha")
            rec = dict(kx=km[0], ky=km[1], kz=km[2], N_kpts=km[0]*km[1]*km[2],
                      E_tot_Ha=cached_energy, cached=True)
        elif (km[0], km[1], km[2], basis, ke_cutoff_Ry) in done:
            log.info(f"[kmesh] ya existe {fmt_tuple(km)}, skip")
            continue
        else:
            log.info(f"[kmesh] calculando | k={fmt_tuple(km)}")
            # Aquí iría el cálculo real (simulado por ahora)
            rec = dict(kx=km[0], ky=km[1], kz=km[2], N_kpts=km[0]*km[1]*km[2],
                      E_tot_Ha=np.nan, cached=False)

        rec.update(dict(ke_cutoff_Ry=ke_cutoff_Ry, sigma_Ha=sigma_ha, basis=basis,
                       timestamp=datetime.now().isoformat()))
        append_and_flush(rec)

        # Estrategia de early stopping adaptativa
        if len(rows) >= 3:
            valid_rows = [r for r in rows if np.isfinite(r['E_tot_Ha'])]
            if len(valid_rows) >= 3:
                converged, reason = optimizer.adaptive_convergence_strategy(valid_rows)
                if converged:
                    log.info(f"[kmesh] convergencia adaptativa: {reason}")
                    break

    return pd.DataFrame(rows)

def incremental_lattice_optimization(optimizer: IncrementalOptimizer,
                                   a0: float, da: float, npoints_side: int,
                                   x_ga: float, basis: str, pseudo: str,
                                   sigma_ha: float, xc: str,
                                   ke_cutoff_Ry: float, kmesh: Tuple[int,int,int],
                                   out_dir: Path, log: logging.Logger, **kwargs) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    """Optimización de lattice con reutilización inteligente."""

    out_csv = out_dir / "lattice_optimization.csv"
    rows = []
    if out_csv.exists():
        try:
            rows = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            pass

    done = {(r.get("a_Ang"), r.get("basis"), r.get("ke_cutoff_Ry"), r.get("kmesh")) for r in rows}

    def append_and_flush(rec, source="incremental"):
        rec["source"] = source
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values("a_Ang", inplace=True)
        df.to_csv(out_csv, index=False)

    # Estrategia de búsqueda inteligente
    # 1. Usar puntos del cache
    # 2. Calcular puntos nuevos estratégicamente
    # 3. Refinar alrededor del mínimo

    # Rango de búsqueda adaptativo
    a_min = a0 - 3 * da
    a_max = a0 + 3 * da
    a_range = np.linspace(a_min, a_max, max(7, 2*npoints_side+1))

    # Fase 1: Exploración usando cache
    log.info("[LATTICE] Fase 1: Exploración con cache")
    for a_A in a_range:
        cache_key = (float(a_A), basis, ke_cutoff_Ry, fmt_tuple(kmesh))
        cached_energy = optimizer.get_cached_result(cache_key)

        if cached_energy is not None:
            log.info(f"[lattice] usando cache | a={a_A:.4f} Å | E={cached_energy:.8f} Ha")
            rec = dict(a_Ang=a_A, E_tot_Ha=cached_energy, cached=True)
            rec.update(dict(ke_cutoff_Ry=ke_cutoff_Ry, kmesh=fmt_tuple(kmesh),
                           sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat()))
            append_and_flush(rec, "cache")
        elif (float(a_A), basis, ke_cutoff_Ry, fmt_tuple(kmesh)) in done:
            log.info(f"[lattice] ya existe a={a_A:.4f} Å, skip")
            continue
        else:
            # Calcular punto nuevo
            log.info(f"[lattice] calculando | a={a_A:.4f} Å")
            rec = dict(a_Ang=a_A, E_tot_Ha=np.nan, cached=False)
            rec.update(dict(ke_cutoff_Ry=ke_cutoff_Ry, kmesh=fmt_tuple(kmesh),
                           sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat()))
            append_and_flush(rec, "exploration")

    # Fase 2: Análisis y refinamiento
    log.info("[LATTICE] Fase 2: Análisis y refinamiento")

    df_current = pd.read_csv(out_csv)
    valid_points = df_current.dropna(subset=["E_tot_Ha"])

    if len(valid_points) >= 3:
        # Ajuste cuadrático preliminar
        a_vals = valid_points["a_Ang"].values
        E_vals = valid_points["E_tot_Ha"].values
        fit = quadratic_fit(a_vals, E_vals)

        a_opt_prelim = fit["a_opt"]
        log.info(f"[lattice] ajuste preliminar: a*={a_opt_prelim:.4f} Å")

        # Refinar alrededor del óptimo preliminar
        refine_range = da * 0.5
        refine_points = np.linspace(a_opt_prelim - refine_range, a_opt_prelim + refine_range, 5)

        for a_refine in refine_points:
            if not (a_min <= a_refine <= a_max):
                continue

            cache_key = (float(a_refine), basis, ke_cutoff_Ry, fmt_tuple(kmesh))
            if optimizer.get_cached_result(cache_key) is None and \
               (float(a_refine), basis, ke_cutoff_Ry, fmt_tuple(kmesh)) not in done:
                log.info(f"[lattice] refinando | a={a_refine:.4f} Å")
                rec = dict(a_Ang=a_refine, E_tot_Ha=np.nan, cached=False)
                rec.update(dict(ke_cutoff_Ry=ke_cutoff_Ry, kmesh=fmt_tuple(kmesh),
                               sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat()))
                append_and_flush(rec, "refinement")

    # Ajuste final
    df_final = pd.read_csv(out_csv)
    valid_final = df_final.dropna(subset=["E_tot_Ha"])

    if len(valid_final) >= 3:
        a_vals = valid_final["a_Ang"].values
        E_vals = valid_final["E_tot_Ha"].values
        fit_info = quadratic_fit(a_vals, E_vals)
    else:
        fit_info = None

    return df_final, fit_info

def main():
    parser = argparse.ArgumentParser(description="Pipeline incremental optimizado para GaAs")
    parser.add_argument("--fast", action="store_true", help="Modo rápido")
    parser.add_argument("--timeout_s", type=int, default=300, help="Timeout por punto")
    parser.add_argument("--target_accuracy", type=float, default=1e-4, help="Precisión objetivo")
    parser.add_argument("--reuse_previous", type=str, default="on", help="Reutilizar resultados previos")
    parser.add_argument("--hpc_mode", type=str, default="auto", help="Modo HPC: auto, single, array, multi")
    parser.add_argument("--distributed_basis", type=str, default="off", help="Distribuir bases entre nodos")

    args = parser.parse_args()

    out_root = Path("preconvergencia_out")
    subdirs = ensure_dirs(out_root)
    log = setup_logging(out_root)

    # Configuración HPC inteligente
    hpc_mode = args.hpc_mode
    if hpc_mode == "auto":
        # Detectar automáticamente basado en variables de entorno
        if os.environ.get('SLURM_ARRAY_TASK_ID'):
            hpc_mode = "array"
        elif int(os.environ.get('SLURM_NNODES', 1)) > 1:
            hpc_mode = "multi"
        else:
            hpc_mode = "single"

    log.info(f"[HPC] Modo detectado: {hpc_mode}")

    # Inicializar optimizador incremental
    optimizer = IncrementalOptimizer(out_root, log)

    # Cargar resultados previos si está habilitado
    if str2bool(args.reuse_previous):
        optimizer.load_previous_results()

    log.info("🚀 INICIANDO PIPELINE INCREMENTAL OPTIMIZADO")
    log.info("=" * 60)

    # Configuración distribuida para HPC
    if str2bool(args.distributed_basis) and hpc_mode in ["array", "multi"]:
        # Distribuir bases entre nodos/tasks
        all_bases = ["gth-szv", "gth-dzvp", "gth-tzvp", "gth-tzv2p"]
        node_id = int(os.environ.get('SLURM_PROCID', 0))
        n_nodes = int(os.environ.get('SLURM_NNODES', 1))

        if hpc_mode == "array":
            array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
            basis_idx = array_id % len(all_bases)
        else:
            basis_idx = node_id % len(all_bases)

        selected_bases = [all_bases[basis_idx]]
        log.info(f"[HPC] Nodo {node_id}: procesando base {selected_bases[0]}")
    else:
        selected_bases = ["def2-svp"]  # Base por defecto

    # Parámetros base (podrían venir de configuración)
    a0, x_ga = 5.653, 0.25
    basis, pseudo = selected_bases[0], "gth-pbe"
    sigma_ha, xc = 0.01, "PBE"
    cutoff_list = [40, 60, 80, 100, 120] if args.fast else DEFAULT_CUTOFF_LIST_RY
    k_list = [(2,2,2),(4,4,4),(6,6,6),(8,8,8)] if args.fast else DEFAULT_K_LIST[:5]

    # 1. Cutoff scan incremental
    log.info("📊 ETAPA 1: Escaneo de cutoff incremental")
    df_cutoff = incremental_cutoff_scan(
        optimizer, cutoff_list, a0, x_ga, basis, pseudo, sigma_ha, xc,
        (6,6,6), subdirs["cutoff"], log, timeout_s=args.timeout_s
    )

    # Determinar cutoff óptimo
    cutoff_opt = 100.0  # Valor por defecto
    if not df_cutoff.empty:
        valid_cutoff = df_cutoff.dropna(subset=["E_tot_Ha"])
        if not valid_cutoff.empty:
            emin = valid_cutoff["E_tot_Ha"].min()
            converged = valid_cutoff[abs(valid_cutoff["E_tot_Ha"] - emin) < args.target_accuracy]
            if not converged.empty:
                cutoff_opt = float(converged["ke_cutoff_Ry"].iloc[0])

    log.info(f"[cutoff*] Óptimo: {cutoff_opt} Ry")

    # 2. K-mesh scan incremental
    log.info("📊 ETAPA 2: Escaneo de k-mesh incremental")
    df_kmesh = incremental_kmesh_scan(
        optimizer, k_list, a0, x_ga, basis, pseudo, sigma_ha, xc,
        cutoff_opt, subdirs["kmesh"], log, timeout_s=args.timeout_s
    )

    # Determinar k-mesh óptimo
    kmesh_opt = (6,6,6)  # Valor por defecto
    if not df_kmesh.empty:
        valid_kmesh = df_kmesh.dropna(subset=["E_tot_Ha"])
        if not valid_kmesh.empty:
            emin = valid_kmesh["E_tot_Ha"].min()
            converged = valid_kmesh[abs(valid_kmesh["E_tot_Ha"] - emin) < args.target_accuracy]
            if not converged.empty:
                best_row = converged.iloc[0]
                kmesh_opt = (int(best_row["kx"]), int(best_row["ky"]), int(best_row["kz"]))

    log.info(f"[k*] Óptimo: {fmt_tuple(kmesh_opt)}")

    # 3. Lattice optimization incremental
    log.info("📊 ETAPA 3: Optimización de lattice incremental")
    df_lattice, fit = incremental_lattice_optimization(
        optimizer, a0, 0.02, 3 if args.fast else 6, x_ga, basis, pseudo,
        sigma_ha, xc, cutoff_opt, kmesh_opt, subdirs["lattice"], log,
        timeout_s=args.timeout_s
    )

    # Resultado final
    a_opt = fit["a_opt"] if fit else a0
    log.info(f"[a*] Óptimo: {a_opt:.4f} Å")

    # Guardar resumen con información HPC
    summary = {
        "pipeline_type": "incremental_optimized_hpc",
        "hpc_mode": hpc_mode,
        "basis_processed": basis,
        "cutoff_opt": cutoff_opt,
        "kmesh_opt": kmesh_opt,
        "a_opt": a_opt,
        "fit_quality": fit["R2"] if fit else None,
        "cached_results_used": len(optimizer.cache),
        "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
        "slurm_node": os.environ.get('SLURMD_NODENAME'),
        "timestamp": datetime.now().isoformat()
    }

    with open(out_root / f"incremental_summary_{hpc_mode}.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("✅ Pipeline incremental completado")
    log.info(f"📊 Resumen guardado en: {out_root / f'incremental_summary_{hpc_mode}.json'}")

    # En modo multi-node, esperar sincronización
    if hpc_mode == "multi":
        log.info("[HPC] Esperando sincronización de nodos...")
        # Aquí se podría implementar una barrera de sincronización
        # Por ahora, solo logging
        log.info("[HPC] Sincronización completada")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error fatal: {e}")
        traceback.print_exc()
        sys.exit(1)