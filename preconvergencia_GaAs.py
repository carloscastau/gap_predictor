#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preconvergencia_GaAs.py  (versión robusta: inline-timeout + progreso + CSV incremental)

- Evita bloqueos por fork+BLAS usando timeout "inline" (signal.alarm) por defecto.
- Loggea progreso por punto (start/ok/timeout) y escribe CSV tras cada evaluación.
- Flujo: cutoff -> kmesh -> E(a) (ajuste cuadrático) -> (opcional) slab y barrido de bases.

Requisitos: pyscf (incl. pyscf.pbc), numpy, pandas, matplotlib, pymatgen, spglib.

Ejemplo exprés (validación en minutos):
  python -u preconvergencia_GaAs.py \
    --fast --nprocs 1 --gpu off --timeout_s 300 --timeout_policy inline \
    --basis_sweep off --basis_list gth-tzv2p,gth-tzvp,gth-dzvp,gth-dzv,gth-szv \
    --sigma_ha 0.01 --cutoff_list 80,120,160 --k_list 2x2x2,4x4x4,6x6x6 \
    --a0 5.653 --da 0.02 --npoints_side 3 --dos off --make_report on
"""

import os
import sys
import json
import math
import argparse
import logging
import traceback
import multiprocessing as mp
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscf import lib as pyscf_lib
from pyscf.pbc import gto, dft
from pyscf import scf as mol_scf
from pyscf.scf import addons as scf_addons

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.bandstructure import HighSymmKpath

# GPU opcional
try:
    import gpu4pyscf  # noqa: F401
    GPU4PYSCF_AVAILABLE = True
except Exception:
    GPU4PYSCF_AVAILABLE = False

# ============================ Constantes/utilidades ============================

HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
RY_TO_HA = 0.5
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

DEFAULT_CUTOFF_LIST_RY = [40, 60, 80, 100, 120, 160, 200]
DEFAULT_K_LIST = [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(8,8,8),(10,10,10),(11,11,11),(12,12,12)] #,(13,13,13)]
UNIVERSAL_GTH_BASES = ["gth-szv","gth-dzv","gth-dzvp","gth-tzvp","gth-tzv2p","gth-qzv2p"]

def setup_logging(out_root: Path) -> logging.Logger:
    log = logging.getLogger("preconv")
    log.setLevel(logging.INFO)
    log.propagate = False
    out_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_root / "preconv.log", mode="a", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh); log.addHandler(ch)
    return log

def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    subdirs = {
        "cutoff": out_root / "cutoff",
        "kmesh": out_root / "kmesh",
        "lattice": out_root / "lattice",
        "bands": out_root / "bands",
        "dos": out_root / "dos",
        "slab": out_root / "slab",
        "bases": out_root / "bases",
        "checkpoints": out_root / "checkpoints",
    }
    for p in subdirs.values(): p.mkdir(parents=True, exist_ok=True)
    return subdirs

def save_checkpoint(out_root: Path, stage: str, data: Dict[str, Any], log: logging.Logger = None):
    """Guarda un checkpoint del progreso actual."""
    checkpoint_dir = out_root / "checkpoints"
    checkpoint_file = checkpoint_dir / f"checkpoint_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    checkpoint_data = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "status": "in_progress"
    }

    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        if log:
            log.info(f"[CHECKPOINT] Guardado: {stage} → {checkpoint_file}")
        return checkpoint_file
    except Exception as e:
        if log:
            log.warning(f"[CHECKPOINT] Error guardando checkpoint {stage}: {e}")
        return None

def save_incremental_checkpoint(out_root: Path, stage: str, data: Dict[str, Any], log: logging.Logger = None):
    """Guarda checkpoint incremental durante cálculos largos."""
    checkpoint_dir = out_root / "checkpoints"
    checkpoint_file = checkpoint_dir / f"checkpoint_{stage}_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    checkpoint_data = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "status": "incremental",
        "type": "incremental"
    }

    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        if log:
            log.info(f"[CHECKPOINT] Incremental guardado: {stage} → {checkpoint_file}")
        return checkpoint_file
    except Exception as e:
        if log:
            log.warning(f"[CHECKPOINT] Error guardando checkpoint incremental {stage}: {e}")
        return None

def load_latest_checkpoint(out_root: Path, stage: str = None) -> Dict[str, Any]:
    """Carga el checkpoint más reciente."""
    checkpoint_dir = out_root / "checkpoints"

    if not checkpoint_dir.exists():
        return {}

    # Buscar checkpoints del stage específico o todos
    pattern = f"checkpoint_{stage}_*.json" if stage else "checkpoint_*.json"
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        return {}

    # Encontrar el más reciente
    latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def parse_k_list(s: str) -> List[Tuple[int,int,int]]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    out = []
    for it in items:
        if "x" in it.lower():
            nx,ny,nz = it.lower().split("x")
            out.append((int(nx),int(ny),int(nz)))
        else:
            n = int(it); out.append((n,n,n))
    return out

def parse_cutoff_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def str2bool(s: str) -> bool:
    return str(s).lower() in ("1","true","on","yes","y")

def ha2ev(x): return float(x) * HARTREE_TO_EV
def ev2ha(x): return float(x) * EV_TO_HARTREE
def ry2ha(x): return float(x) * RY_TO_HA
def fmt_tuple(t): return f"{t[0]}x{t[1]}x{t[2]}"

# --------- Verificación ligera de bases (sin PBC) ---------
from pyscf import gto as mol_gto
from pyscf.gto import basis as basis_mod
try:
    from pyscf.gto import pseudo as pseudo_mod
except Exception:
    pseudo_mod = None

def basis_available_for_elements(basis: str, elements=("Ga","As"), pseudo="gth-pbe") -> bool:
    try:
        for el in set(elements):
            basis_mod.load(basis, el)
        if pseudo and ("gth" in pseudo.lower()) and (pseudo_mod is not None):
            for el in set(elements):
                pseudo_mod.load(pseudo, el)
        mol = mol_gto.Mole()
        mol.atom = "\n".join([f"{el} 0 0 0" for el in set(elements)])
        mol.basis = {el: basis for el in elements}
        if pseudo and ("gth" in pseudo.lower()):
            mol.ecp = {el: pseudo for el in elements}
        mol.build(verbose=0)
        return True
    except Exception:
        return False

def filter_available_gth_bases(candidate_bases: List[str], log: logging.Logger) -> List[str]:
    avail = []
    for b in candidate_bases:
        if "gth" not in b.lower():
            log.warning(f"[bases] '{b}' no es familia GTH. Se ignora.")
            continue
        if basis_available_for_elements(b):
            avail.append(b)
        else:
            log.warning(f"[bases] '{b}' no está disponible para Ga/As en esta instalación. Se ignora.")
    if not avail:
        log.warning("[bases] No se pudo verificar ninguna base; usando fallback 'gth-dzvp'.")
        avail = ["gth-dzvp"]
    return avail

# ============================ Celdas y SCF ============================

def build_gaas_cell(a_A: float, x_ga: float, basis: str, pseudo: str, ke_cutoff_Ry: float,
                     precision: float = 1e-8, verbose: int = 0, log: logging.Logger = None) -> gto.Cell:
    # Definición corregida de la celda unitaria FCC para GaAs (estructura zincblende)
    a1 = np.array([a_A, 0.0, 0.0])
    a2 = np.array([0.0, a_A, 0.0])
    a3 = np.array([0.0, 0.0, a_A])

    # Posición del átomo de Ga en coordenadas fraccionarias (x_ga, x_ga, x_ga)
    # Para estructura zincblende estándar, x_ga = 0.25
    r_ga_frac = np.array([x_ga, x_ga, x_ga])
    r_ga_cart = r_ga_frac @ np.vstack([a1, a2, a3])

    cell = gto.Cell()
    cell.unit = "A"
    cell.a = np.vstack([a1, a2, a3])
    cell.atom = [("As", (0.0, 0.0, 0.0)), ("Ga", (float(r_ga_cart[0]), float(r_ga_cart[1]), float(r_ga_cart[2])))]
    cell.basis = {"Ga": basis, "As": basis}

    # Usar pseudopotenciales solo si están disponibles
    try:
        cell.pseudo = {"Ga": pseudo, "As": pseudo}
        if log: log.info(f"[CELL] Usando pseudopotenciales: {pseudo}")
    except Exception:
        if log: log.warning(f"[CELL] Pseudopotenciales {pseudo} no disponibles, usando all-electron")
        cell.pseudo = None

    cell.ke_cutoff = ry2ha(ke_cutoff_Ry)
    cell.precision = precision
    cell.verbose = verbose
    cell.exp_to_discard = 0.1

    # Agregar verificación de validez de la estructura
    cell.build()

    # Verificación adicional: asegurar que los átomos no estén en el mismo lugar
    if np.allclose([0.0, 0.0, 0.0], [r_ga_cart[0], r_ga_cart[1], r_ga_cart[2]], atol=1e-10):
        raise ValueError(f"¡ERROR! Átomos Ga y As están en la misma posición. x_ga={x_ga} podría ser 0.")

    return cell

def zeroT_energy_from_smearing(mf) -> float:
    e_tot = getattr(mf, "e_tot", np.nan)
    e_free = getattr(mf, "e_free", None)
    return 0.5*(e_tot+e_free) if e_free is not None else e_tot

def make_kpts(cell: gto.Cell, kmesh: Tuple[int,int,int]) -> np.ndarray:
    kpts = cell.make_kpts(list(kmesh))

    # Verificación adicional para asegurar que los k-points estén correctamente distribuidos
    if len(kpts) == 0:
        raise ValueError(f"No se pudieron generar k-points para la malla {kmesh}")

    # Log de información útil sobre los k-points
    kpt_distances = []
    for i in range(1, len(kpts)):
        dist = np.linalg.norm(kpts[i] - kpts[i-1])
        kpt_distances.append(dist)

    min_dist = min(kpt_distances) if kpt_distances else 0
    max_dist = max(kpt_distances) if kpt_distances else 0

    print(f"[KPTS] Malla {kmesh} → {len(kpts)} puntos, Δk_min={min_dist:.4f}, Δk_max={max_dist:.4f} (2π/Å)")

    return kpts

def run_scf(cell: gto.Cell, kmesh: Tuple[int,int,int], xc: str, sigma_ha: float,
            log: logging.Logger = None, max_cycle: int = 80, conv_tol: float = 1e-8,
            n_retries: int = 2, gpu: bool = False):
    # Verificar que la celda se construyó correctamente
    if not hasattr(cell, 'a') or cell.a is None:
        raise ValueError("La celda no está construida correctamente")
    kpts = make_kpts(cell, kmesh)
    kmf = dft.KRKS(cell, kpts=kpts)
    kmf.xc = xc
    kmf.conv_tol = conv_tol
    kmf.max_cycle = max_cycle
    kmf = scf_addons.smearing_(kmf, sigma=sigma_ha, method="fermi")
    kmf = mol_scf.addons.remove_linear_dep_(kmf)
    if gpu and GPU4PYSCF_AVAILABLE:
        try:
            kmf = kmf.to_gpu()
            if log: log.info("[GPU] KRKS → GPU")
        except Exception as e:
            if log: log.warning(f"[GPU] No GPU4PySCF ({e}). CPU.")
    ok = False; exc = None
    for attempt in range(n_retries+1):
        try:
            e = kmf.kernel()
            if kmf.converged:
                ok = True; break
            else:
                if log: log.warning(f"[SCF] Intento {attempt+1} no convergió. e_tot={e}")
        except Exception as ex:
            exc = ex
            if log: log.warning(f"[SCF] Excepción intento {attempt+1}: {ex}")
        kmf.max_cycle = max(100, kmf.max_cycle + 20)
        kmf.level_shift = 0.1; kmf.damp = 0.5; kmf.diis_space = 12
    if not ok and exc and log:
        log.error(f"[SCF] Fallo definitivo: {exc}")
    return kmf, ok

# ============================ Timeouts (inline/subproc) ============================

class _TimeoutExc(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _TimeoutExc()

def _run_point_core(a_A, x_ga, basis, pseudo, ke, kmesh, xc, sigma, conv, gpu, log=None):
    cell = build_gaas_cell(a_A, x_ga, basis, pseudo, ke, log=log)
    kmf, ok = run_scf(cell, kmesh, xc, sigma, log=None, gpu=gpu, conv_tol=conv)
    e0 = zeroT_energy_from_smearing(kmf) if ok else np.nan
    nk = len(make_kpts(cell, kmesh))
    return ok, e0, nk

def run_point_inline_timeout(args, timeout_s: int):
    a_A,x_ga,basis,pseudo,ke,kmesh,xc,sigma,conv,gpu,log = args
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return _run_point_core(a_A,x_ga,basis,pseudo,ke,kmesh,xc,sigma,conv,gpu)
    except _TimeoutExc:
        return False, np.nan, 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def run_point_subproc_timeout(args, timeout_s: int):
    q = mp.Queue()
    def _wrap(q, A):
        try: q.put(_run_point_core(*A))
        except Exception: q.put((False, np.nan, 0))
    p = mp.Process(target=_wrap, args=(q,args))
    p.start(); p.join(timeout_s)
    if p.is_alive():
        p.terminate(); p.join()
        return False, np.nan, 0
    return q.get()

def run_with_policy(args, timeout_s: int, policy: str):
    if timeout_s is None or timeout_s <= 0:
        return _run_point_core(*args)
    if policy == "subproc":
        return run_point_subproc_timeout(args, timeout_s)
    # por defecto: inline (seguro para BLAS)
    return run_point_inline_timeout(args, timeout_s)

# ============================ Workers (usados con map o en serie) ============================

def _task_cutoff(cut_Ry, a_A, x_ga, basis, pseudo, sigma_ha, xc, kmesh_fixed, conv_tol, gpu, timeout_s, policy, log=None, checkpoint_callback=None):
    ok, e0, nk = run_with_policy((a_A, x_ga, basis, pseudo, cut_Ry, kmesh_fixed, xc, sigma_ha, conv_tol, gpu, log),
                                   timeout_s, policy)

    result = dict(ke_cutoff_Ry=cut_Ry, E_tot_Ha=e0, n_kpts=nk)

    # Guardar checkpoint incremental si se proporciona callback
    if checkpoint_callback and ok:
        checkpoint_callback("cutoff_point", {"cutoff_Ry": cut_Ry, "energy_Ha": e0, "n_kpts": nk})

    return result

def _task_kmesh(kmesh, a_A, x_ga, basis, pseudo, sigma_ha, xc, ke_cutoff_Ry, conv_tol, gpu, timeout_s, policy, log=None, checkpoint_callback=None):
    ok, e0, nk = run_with_policy((a_A, x_ga, basis, pseudo, ke_cutoff_Ry, kmesh, xc, sigma_ha, conv_tol, gpu, log),
                                   timeout_s, policy)

    result = dict(kx=kmesh[0], ky=kmesh[1], kz=kmesh[2], N_kpts=nk, E_tot_Ha=e0)

    # Guardar checkpoint incremental si se proporciona callback
    if checkpoint_callback and ok:
        checkpoint_callback("kmesh_point", {"kmesh": kmesh, "energy_Ha": e0, "n_kpts": nk})

    return result

def _task_lattice(a_A, x_ga, basis, pseudo, sigma_ha, xc, ke_cutoff_Ry, kmesh, conv_tol, gpu, timeout_s, policy, log=None, checkpoint_callback=None):
    ok, e0, nk = run_with_policy((a_A, x_ga, basis, pseudo, ke_cutoff_Ry, kmesh, xc, sigma_ha, conv_tol, gpu, log),
                                   timeout_s, policy)

    result = dict(a_Ang=a_A, E_tot_Ha=e0)

    # Guardar checkpoint incremental si se proporciona callback
    if checkpoint_callback and ok:
        checkpoint_callback("lattice_point", {"a_Ang": a_A, "energy_Ha": e0, "n_kpts": nk})

    return result

# ============================ Etapa 1: cutoff (con progreso y CSV incremental) ============================

def scan_cutoff(a_A: float, x_ga: float, basis: str, pseudo: str,
                sigma_ha: float, xc: str,
                cutoff_list_Ry: List[float],
                kmesh_fixed: Tuple[int,int,int],
                out_dir: Path, log: logging.Logger,
                nprocs: int = 1, gpu: bool = False,
                conv_tol: float = 1e-8, timeout_s:int = 300, policy: str = "inline") -> pd.DataFrame:
    out_csv = out_dir / "cutoff.csv"
    # Cargar lo previo
    rows = []
    if out_csv.exists():
        try: rows = pd.read_csv(out_csv).to_dict("records")
        except Exception: pass
    done = {(r.get("ke_cutoff_Ry"), r.get("kmesh"), r.get("basis")) for r in rows}

    def append_and_flush(rec):
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values("ke_cutoff_Ry", inplace=True)
        df.to_csv(out_csv, index=False)

    # Callback para checkpoints incrementales
    def checkpoint_callback(stage, data):
        incremental_data = {
            "basis": basis,
            "kmesh_fixed": kmesh_fixed,
            "a_A": a_A,
            "x_ga": x_ga,
            "sigma_ha": sigma_ha,
            "xc": xc,
            "conv_tol": conv_tol,
            "gpu": gpu,
            "timeout_s": timeout_s,
            "policy": policy,
            **data
        }
        save_incremental_checkpoint(out_dir.parent, stage, incremental_data, log)

    # Loop secuencial para progreso claro (normalmente son pocos puntos)
    for cut in cutoff_list_Ry:
        if (cut, fmt_tuple(kmesh_fixed), basis) in done:
            log.info(f"[cutoff] ya existe cut={cut} Ry, skip")
            continue
        log.info(f"[cutoff] start | basis={basis} | k={fmt_tuple(kmesh_fixed)} | a={a_A:.4f} Å | σ={sigma_ha} Ha | cut={cut} Ry")
        rec = _task_cutoff(cut, a_A, x_ga, basis, pseudo, sigma_ha, xc, kmesh_fixed, conv_tol, gpu, timeout_s, policy, log, checkpoint_callback)
        rec.update(dict(kmesh=fmt_tuple(kmesh_fixed), sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat()))
        if np.isfinite(rec["E_tot_Ha"]):
            log.info(f"[cutoff] ok   | cut={cut} Ry | E={rec['E_tot_Ha']:.8f} Ha | Nk={rec['n_kpts']}")
        else:
            log.warning(f"[cutoff] timeout/fallo | cut={cut} Ry → NaN")
        append_and_flush(rec)

    # Figura
    try:
        df = pd.read_csv(out_csv)
        if not df.empty:
            fig, ax = plt.subplots(figsize=(7,4.5))
            ax.plot(df["ke_cutoff_Ry"], df["E_tot_Ha"], marker="o", lw=1.5)
            ax.set_xlabel("ke_cutoff (Ry)")
            ax.set_ylabel("E (Ha)  [0 K ≈ (E_tot + E_free)/2]")
            ax.grid(True, ls="--", alpha=0.5)
            ax.set_title(f"E vs cutoff | basis={basis}, k={fmt_tuple(kmesh_fixed)}, σ={sigma_ha} Ha, a={a_A:.4f} Å")
            fig.tight_layout(); fig.savefig(out_dir / "E_vs_cutoff.png", dpi=180); plt.close(fig)
        return df
    except Exception:
        return pd.DataFrame(rows)

def choose_cutoff(df: pd.DataFrame, thr_Ha: float = 1e-4) -> float:
    if df is None or df.empty or df["E_tot_Ha"].isna().all(): return None
    emin = df["E_tot_Ha"].min()
    df2 = df.dropna().sort_values("ke_cutoff_Ry")
    for _, r in df2.iterrows():
        if abs(r["E_tot_Ha"] - emin) <= thr_Ha: return float(r["ke_cutoff_Ry"])
    return float(df2["ke_cutoff_Ry"].iloc[-1])

# ============================ Etapa 2: k-mesh (progreso + CSV incremental) ============================

def scan_kmesh(a_A: float, x_ga: float, basis: str, pseudo: str,
               sigma_ha: float, xc: str,
               ke_cutoff_Ry: float, k_list: List[Tuple[int,int,int]],
               out_dir: Path, log: logging.Logger,
               nprocs: int = 1, gpu: bool = False,
               conv_tol: float = 1e-8,
               early_stop_tol_Ha: float = 1e-5,
               timeout_s:int = 300, policy: str = "inline") -> pd.DataFrame:
    out_csv = out_dir / "kmesh.csv"
    rows = []
    if out_csv.exists():
        try: rows = pd.read_csv(out_csv).to_dict("records")
        except Exception: pass
    done = {(r.get("kx"), r.get("ky"), r.get("kz"), r.get("basis"), r.get("ke_cutoff_Ry")) for r in rows}

    def append_and_flush(rec):
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values(["kx","ky","kz"], inplace=True)
        df.to_csv(out_csv, index=False)

    # Callback para checkpoints incrementales
    def checkpoint_callback(stage, data):
        incremental_data = {
            "basis": basis,
            "ke_cutoff_Ry": ke_cutoff_Ry,
            "a_A": a_A,
            "x_ga": x_ga,
            "sigma_ha": sigma_ha,
            "xc": xc,
            "conv_tol": conv_tol,
            "gpu": gpu,
            "timeout_s": timeout_s,
            "policy": policy,
            **data
        }
        save_incremental_checkpoint(out_dir.parent, stage, incremental_data, log)

    energies = []; last_e = None; consecutive_small = 0
    for km in k_list:
        if (km[0],km[1],km[2],basis,ke_cutoff_Ry) in done:
            log.info(f"[kmesh] ya existe {fmt_tuple(km)}, skip")
            continue
        log.info(f"[kmesh] start | basis={basis} | cut={ke_cutoff_Ry} Ry | k={fmt_tuple(km)} | a={a_A:.4f} Å")
        rec = _task_kmesh(km, a_A, x_ga, basis, pseudo, sigma_ha, xc, ke_cutoff_Ry, conv_tol, gpu, timeout_s, policy, log, checkpoint_callback)
        rec.update(dict(ke_cutoff_Ry=ke_cutoff_Ry, sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat()))
        append_and_flush(rec)
        if np.isfinite(rec["E_tot_Ha"]):
            log.info(f"[kmesh] ok   | k={fmt_tuple(km)} | E={rec['E_tot_Ha']:.8f} Ha | Nk={rec['N_kpts']}")
        else:
            log.warning(f"[kmesh] timeout/fallo | k={fmt_tuple(km)} → NaN")
        energies.append(rec["E_tot_Ha"])
        if last_e is not None and np.isfinite(rec["E_tot_Ha"]) and np.isfinite(last_e):
            consecutive_small = consecutive_small + 1 if abs(rec["E_tot_Ha"] - last_e) < early_stop_tol_Ha else 0
        last_e = rec["E_tot_Ha"]
        if consecutive_small >= 2:
            log.info("[kmesh] early-stopping: 2 pasos con ΔE < umbral")
            break

    # Figura
    try:
        df = pd.read_csv(out_csv)
        if not df.empty:
            dfp = df.dropna(subset=["E_tot_Ha"]).copy()
            if not dfp.empty:
                dfp["label"] = dfp.apply(lambda r: f'{int(r["kx"])}x{int(r["ky"])}x{int(r["kz"])}', axis=1)
                dfp["Nk"] = dfp["N_kpts"]
                fig, ax = plt.subplots(figsize=(7,4.5))
                ax.plot(dfp["Nk"], dfp["E_tot_Ha"], marker="o", lw=1.5)
                for _, rr in dfp.iterrows():
                    ax.annotate(rr["label"], (rr["Nk"], rr["E_tot_Ha"]), textcoords="offset points", xytext=(5,5), fontsize=8)
                ax.set_xlabel(r"$N_k = k_x k_y k_z$")
                ax.set_ylabel("E (Ha)  [0 K ≈ (E_tot + E_free)/2]")
                ax.grid(True, ls="--", alpha=0.5)
                ax.set_title(f"E vs k-mesh | basis={basis}, cutoff={ke_cutoff_Ry} Ry, σ={sigma_ha} Ha, a={a_A:.4f} Å")
                fig.tight_layout(); fig.savefig(out_dir / "E_vs_kmesh.png", dpi=180); plt.close(fig)
        return df
    except Exception:
        return pd.DataFrame(rows)

def choose_kmesh(df: pd.DataFrame, thr_Ha: float = 1e-5) -> Tuple[int,int,int]:
    if df is None or df.empty or df["E_tot_Ha"].isna().all(): return None
    emin = df["E_tot_Ha"].min()
    df2 = df.dropna().copy()
    df2["delta"] = (df2["E_tot_Ha"] - emin).abs()
    df2.sort_values(["delta", "N_kpts"], inplace=True)
    r = df2.iloc[0]
    return (int(r["kx"]), int(r["ky"]), int(r["kz"]))

# ============================ Etapa 3: E(a) + ajuste (progreso + CSV incremental) ============================

def quadratic_fit(a_vals: np.ndarray, E_vals: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(a_vals) & np.isfinite(E_vals)
    x = np.array(a_vals)[mask]; y = np.array(E_vals)[mask]
    coeffs, cov = np.polyfit(x, y, 2, cov=True)
    A, B, C = coeffs
    a_opt = -B/(2*A); E_min = np.polyval(coeffs, a_opt)
    yfit = np.polyval(coeffs, x)
    ss_res = np.sum((y - yfit)**2); ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    dA = B/(2*A*A); dB = -1.0/(2*A)
    varA = cov[0,0]; varB = cov[1,1]; covAB = cov[0,1]
    var_aopt = dA*dA*varA + dB*dB*varB + 2*dA*dB*covAB
    sigma_aopt = math.sqrt(abs(var_aopt)) if var_aopt>=0 else np.nan
    return dict(A=A, B=B, C=C, a_opt=a_opt, E_min=E_min, R2=R2, cov=cov, sigma_aopt=sigma_aopt)

def advanced_lattice_optimization(a0: float, da: float, npoints_side: int,
                                  x_ga: float, basis: str, pseudo: str,
                                  sigma_ha: float, xc: str,
                                  ke_cutoff_Ry: float, kmesh: Tuple[int,int,int],
                                  out_dir: Path, log: logging.Logger,
                                  nprocs: int = 1, gpu: bool = False,
                                  conv_tol: float = 1e-8,
                                  max_refine: int = 2,
                                  n_random_restarts: int = 3,
                                  timeout_s: int = 300, policy: str = "inline",
                                  enable_multi_start: bool = True) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    """
    Optimización avanzada de geometría con técnicas de búsqueda global.

    Estrategias implementadas:
    1. Multi-start: Múltiples puntos de inicio aleatorios
    2. Refinamiento local: Búsqueda local alrededor de mínimos candidatos
    3. Detección de mínimos locales vs globales
    4. Estrategia de escape de mínimos locales
    """

    out_csv = out_dir / "lattice_optimization.csv"
    out_summary_csv = out_dir / "optimization_summary.csv"

    # Cargar datos previos si existen
    rows = []
    if out_csv.exists():
        try: rows = pd.read_csv(out_csv).to_dict("records")
        except Exception: pass

    done = {(r.get("a_Ang"), r.get("basis"), r.get("ke_cutoff_Ry"), r.get("kmesh")) for r in rows}

    def append_and_flush(rec, source="main"):
        rec["source"] = source
        rows.append(rec)
        df = pd.DataFrame(rows)
        df.sort_values("a_Ang", inplace=True)
        df.to_csv(out_csv, index=False)

    # Callback para checkpoints incrementales
    def checkpoint_callback(stage, data):
        incremental_data = {
            "basis": basis,
            "ke_cutoff_Ry": ke_cutoff_Ry,
            "kmesh": kmesh,
            "x_ga": x_ga,
            "sigma_ha": sigma_ha,
            "xc": xc,
            "conv_tol": conv_tol,
            "gpu": gpu,
            "timeout_s": timeout_s,
            "policy": policy,
            **data
        }
        save_incremental_checkpoint(out_dir.parent, stage, incremental_data, log)

    # Función para evaluar energía en un punto
    def evaluate_point(a_A, source="main"):
        key = (float(a_A), basis, ke_cutoff_Ry, fmt_tuple(kmesh))
        if key in done:
            log.info(f"[lattice] ya existe a={a_A:.4f} Å, skip")
            return None

        log.info(f"[lattice] start | a={a_A:.4f} Å | basis={basis} | cut={ke_cutoff_Ry} Ry | k={fmt_tuple(kmesh)} | fuente={source}")
        rec = _task_lattice(a_A, x_ga, basis, pseudo, sigma_ha, xc, ke_cutoff_Ry, kmesh, conv_tol, gpu, timeout_s, policy, log, checkpoint_callback)
        rec.update(dict(a_Ang=a_A, ke_cutoff_Ry=ke_cutoff_Ry, kmesh=fmt_tuple(kmesh),
                        sigma_Ha=sigma_ha, basis=basis, timestamp=datetime.now().isoformat(), source=source))
        append_and_flush(rec, source)

        if np.isfinite(rec["E_tot_Ha"]):
            log.info(f"[lattice] ok   | a={a_A:.4f} Å | E={rec['E_tot_Ha']:.8f} Ha | fuente={source}")
            return rec["E_tot_Ha"]
        else:
            log.warning(f"[lattice] timeout/fallo | a={a_A:.4f} Å → NaN | fuente={source}")
            return np.nan

    # 1. FASE 1: Exploración inicial amplia
    log.info("🔍 FASE 1: Exploración inicial amplia")
    initial_range = da * max(3, npoints_side)  # Rango más amplio para exploración
    initial_points = np.linspace(a0 - initial_range, a0 + initial_range, max(7, 2*npoints_side+1))

    exploration_energies = []
    for a_A in initial_points:
        energy = evaluate_point(a_A, "exploracion")
        if np.isfinite(energy):
            exploration_energies.append((a_A, energy))

    # 2. FASE 2: Multi-start con puntos aleatorios
    if enable_multi_start and n_random_restarts > 0:
        log.info(f"🎲 FASE 2: Multi-start con {n_random_restarts} reinicios aleatorios")

        # Encontrar el mejor punto de la exploración inicial
        if exploration_energies:
            exploration_energies.sort(key=lambda x: x[1])
            best_a_initial = exploration_energies[0][0]
            best_E_initial = exploration_energies[0][1]

            # Generar puntos de reinicio aleatorios alrededor del mejor encontrado
            np.random.seed(42)  # Para reproducibilidad
            random_starts = []

            for i in range(n_random_restarts):
                # Estrategia: puntos aleatorios en un rango adaptativo
                adaptive_range = da * (3 - i * 0.5)  # Rango decreciente
                a_random = np.random.uniform(best_a_initial - adaptive_range, best_a_initial + adaptive_range)
                a_random = max(5.0, min(6.5, a_random))  # Mantener en rango físico
                random_starts.append(a_random)

            # Evaluar puntos de reinicio
            for a_random in random_starts:
                evaluate_point(a_random, "random_restart")

    # 3. FASE 3: Refinamiento local alrededor de mínimos candidatos
    log.info("🔧 FASE 3: Refinamiento local")

    # Cargar todos los datos hasta ahora
    df_current = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)
    valid_points = df_current.dropna(subset=["E_tot_Ha"])

    if not valid_points.empty:
        # Encontrar múltiples candidatos a mínimos locales
        a_vals = valid_points["a_Ang"].values
        E_vals = valid_points["E_tot_Ha"].values

        # Ordenar por energía
        sorted_indices = np.argsort(E_vals)
        candidate_as = a_vals[sorted_indices[:3]]  # Top 3 candidatos

        for candidate_a in candidate_as:
            # Refinamiento local alrededor de cada candidato
            local_range = da * 0.5  # Rango más pequeño para refinamiento
            local_points = np.linspace(candidate_a - local_range, candidate_a + local_range, 5)

            for a_local in local_points:
                if not np.isfinite(a_local): continue
                evaluate_point(a_local, "refinamiento")

    # 4. FASE 4: Análisis final y ajuste
    log.info("📊 FASE 4: Análisis final")

    try:
        df_final = pd.read_csv(out_csv)
        if df_final.empty:
            return pd.DataFrame(rows), None

        # Análisis estadístico de los resultados
        valid_df = df_final.dropna(subset=["E_tot_Ha"])
        if valid_df.empty:
            log.warning("No hay puntos válidos para el análisis final")
            return df_final, None

        # Estadísticas descriptivas
        E_stats = valid_df["E_tot_Ha"].describe()
        a_stats = valid_df["a_Ang"].describe()

        log.info(f"[OPT] Estadísticas: E_min={E_stats['min']:.6f}, E_max={E_stats['max']:.6f}")
        log.info(f"[OPT] Estadísticas: a_min={a_stats['min']:.4f}, a_max={a_stats['max']:.4f}")

        # Detectar posibles mínimos locales
        a_vals = valid_df["a_Ang"].values
        E_vals = valid_df["E_tot_Ha"].values

        # Ordenar por posición
        sort_idx = np.argsort(a_vals)
        a_sorted = a_vals[sort_idx]
        E_sorted = E_vals[sort_idx]

        # Detectar cambios de pendiente (posibles mínimos)
        dE_da = np.gradient(E_sorted, a_sorted)
        possible_minima = []

        for i in range(1, len(dE_da)-1):
            if dE_da[i-1] < 0 and dE_da[i+1] > 0:  # Cambio de pendiente negativo a positivo
                possible_minima.append((a_sorted[i], E_sorted[i]))

        log.info(f"[OPT] Detectados {len(possible_minima)} posibles mínimos locales")

        # Si hay múltiples mínimos, evaluar cuál es el global
        if len(possible_minima) > 1:
            possible_minima.sort(key=lambda x: x[1])  # Ordenar por energía
            best_minimum = possible_minima[0]
            log.info(f"[OPT] Mínimo global candidato: a={best_minimum[0]:.4f} Å, E={best_minimum[1]:.6f} Ha")

            # Estrategia de escape: si el mínimo parece sospechoso, explorar más
            if len(possible_minima) > 2:
                log.info("[OPT] Múltiples mínimos detectados - ejecutando verificación adicional")
                # Agregar puntos entre mínimos para verificar
                for i in range(len(possible_minima)-1):
                    a_mid = (possible_minima[i][0] + possible_minima[i+1][0]) / 2
                    evaluate_point(a_mid, "verificacion")

        # Ajuste cuadrático final
        fit_info = quadratic_fit(a_vals, E_vals)
        A, a_opt, R2 = fit_info["A"], fit_info["a_opt"], fit_info["R2"]

        log.info(f"[OPT] Ajuste final: A={A:.4e}, a*={a_opt:.6f} Å, R²={R2:.6f}")

        # Validación de calidad del ajuste
        if R2 < 0.95:
            log.warning(f"[OPT] Calidad de ajuste baja: R²={R2:.4f}. Considere más puntos.")
        elif R2 > 0.99:
            log.info(f"[OPT] Excelente calidad de ajuste: R²={R2:.4f}")

        # Crear resumen de optimización
        summary_data = {
            'optimization_method': 'multi_start_advanced',
            'n_total_points': len(df_final),
            'n_valid_points': len(valid_df),
            'n_possible_minima': len(possible_minima),
            'best_a_candidate': best_minimum[0] if len(possible_minima) > 1 else a_opt,
            'best_E_candidate': best_minimum[1] if len(possible_minima) > 1 else fit_info['E_min'],
            'quadratic_a_opt': a_opt,
            'quadratic_E_min': fit_info['E_min'],
            'fit_R2': R2,
            'fit_sigma_a': fit_info.get('sigma_aopt', np.nan),
            'energy_range': E_stats['max'] - E_stats['min'],
            'a_range': a_stats['max'] - a_stats['min'],
            'timestamp': datetime.now().isoformat()
        }

        pd.DataFrame([summary_data]).to_csv(out_summary_csv, index=False)

        # Plot mejorado con información de optimización
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot principal
        ax1.plot(df_final["a_Ang"], df_final["E_tot_Ha"], "o", label="Datos", alpha=0.7)
        a_line = np.linspace(df_final["a_Ang"].min(), df_final["a_Ang"].max(), 200)
        E_line = np.polyval([fit_info["A"], fit_info["B"], fit_info["C"]], a_line)
        ax1.plot(a_line, E_line, "-", label=f"Ajuste cuadrático\nR²={R2:.4f}")
        ax1.axvline(a_opt, ls="--", color="red", alpha=0.7, label=f"a_opt={a_opt:.4f} Å")
        ax1.set_ylabel("E (Ha)"); ax1.grid(True, ls="--", alpha=0.5)
        ax1.set_title("Optimización Avanzada de Lattice")
        ax1.legend()

        # Residuos
        yfit_pts = np.polyval([fit_info["A"], fit_info["B"], fit_info["C"]], df_final["a_Ang"].values)
        resid = df_final["E_tot_Ha"].values - yfit_pts
        ax2.axhline(0.0, color="k", lw=1); ax2.plot(df_final["a_Ang"], resid, "o-")
        ax2.set_xlabel("a (Å)"); ax2.set_ylabel("Residuos (Ha)")
        ax2.set_title("Análisis de Residuos")
        ax2.grid(True, ls="--", alpha=0.5)

        # Información de fuentes
        source_colors = {'exploracion': 'blue', 'random_restart': 'red', 'refinamiento': 'green', 'verificacion': 'orange'}
        for source, color in source_colors.items():
            source_data = df_final[df_final['source'] == source]
            if not source_data.empty:
                ax3.scatter(source_data['a_Ang'], source_data['E_tot_Ha'],
                           c=color, label=source, alpha=0.7, s=30)

        ax3.set_xlabel("a (Å)"); ax3.set_ylabel("E (Ha)")
        ax3.set_title("Puntos por Fuente")
        ax3.legend(); ax3.grid(True, ls="--", alpha=0.5)

        # Histograma de energías
        ax4.hist(valid_df["E_tot_Ha"], bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(fit_info['E_min'], color='red', ls='--', label=f'E_min={fit_info["E_min"]:.6f}')
        ax4.set_xlabel("E (Ha)"); ax4.set_ylabel("Frecuencia")
        ax4.set_title("Distribución de Energías")
        ax4.legend()

        fig.suptitle(f"Optimización Avanzada - Basis: {basis}, Cutoff: {ke_cutoff_Ry} Ry, k: {fmt_tuple(kmesh)}", fontsize=14)
        fig.tight_layout(); fig.savefig(out_dir / "advanced_optimization.png", dpi=180, bbox_inches='tight'); plt.close(fig)

        return df_final, fit_info

    except Exception as e:
        log.error(f"[OPT] Error en análisis final: {e}")
        return pd.DataFrame(rows), None

# ============================ Bandas/DOS/Slab (sin cambios funcionales) ============================

def build_pmg_structure(a_A: float, x_ga: float) -> Structure:
    a1 = [0.0, a_A/2.0, a_A/2.0]; a2 = [a_A/2.0, 0.0, a_A/2.0]; a3 = [a_A/2.0, a_A/2.0, 0.0]
    lattice = Lattice([a1,a2,a3])
    species = ["As","Ga"]; frac_coords = [[0,0,0], [x_ga, x_ga, x_ga]]
    return Structure(lattice, species, frac_coords, coords_are_cartesian=False)

def get_hs_kpath(structure: Structure, line_density: int = 40) -> Tuple[np.ndarray, List[str], List[int]]:
    kpath = HighSymmKpath(structure)
    kpts_labels = kpath.kpath["kpoints"]; path = kpath.kpath["path"]
    kpts_frac = []; labels = []; segment_breaks = []
    for seg in path:
        start, end = seg
        k0 = np.array(kpts_labels[start]); k1 = np.array(kpts_labels[end])
        for i in range(line_density):
            t = i/(line_density-1)
            kpts_frac.append((1-t)*k0 + t*k1); labels.append("")
        labels[-line_density] = start; labels[-1] = end
        segment_breaks.append(len(kpts_frac)-1)
    return np.array(kpts_frac), labels, segment_breaks

def kpath_distances(cell: gto.Cell, kpts_frac: np.ndarray) -> np.ndarray:
    b = cell.reciprocal_vectors(); k_cart = (kpts_frac @ b)
    d = np.zeros(len(k_cart))
    for i in range(1, len(k_cart)):
        d[i] = d[i-1] + np.linalg.norm(k_cart[i] - k_cart[i-1])
    return d

def extract_gap_from_kmf(kmf) -> Dict[str, Any]:
    mo_e = getattr(kmf, "mo_energy", None); mo_occ = getattr(kmf, "mo_occ", None)
    if mo_e is None or mo_occ is None:
        print("[GAP] Error: No se pudieron obtener mo_energy o mo_occ del objeto kmf")
        return dict(gap_eV=np.nan, direct=False, k_VBM=None, k_CBM=None, E_VBM_eV=np.nan, E_CBM_eV=np.nan, mu_eV=np.nan)

    e_vbm = -1e9; e_cbm = 1e9; idx_vbm = (None, None); idx_cbm = (None, None)
    n_valence = 0; n_conduction = 0

    for ik, (ek, occk) in enumerate(zip(mo_e, mo_occ)):
        for ib, (e, occ) in enumerate(zip(ek, occk)):
            if occ > 0.5:  # Banda de valencia
                n_valence += 1
                if e > e_vbm:
                    e_vbm = float(e)
                    idx_vbm = (ik, ib)
            elif occ <= 0.5:  # Banda de conducción
                n_conduction += 1
                if e < e_cbm:
                    e_cbm = float(e)
                    idx_cbm = (ik, ib)

    gap_ha = max(0.0, e_cbm - e_vbm)
    mu_ha = 0.5 * (e_cbm + e_vbm) if e_vbm > -1e9 and e_cbm < 1e9 else 0.0

    gap_info = dict(
        gap_eV=ha2ev(gap_ha),
        direct=(idx_vbm[0]==idx_cbm[0]),
        k_VBM=idx_vbm[0],
        k_CBM=idx_cbm[0],
        E_VBM_eV=ha2ev(e_vbm),
        E_CBM_eV=ha2ev(e_cbm),
        mu_eV=ha2ev(mu_ha)
    )

    # Logging detallado para diagnóstico
    print(f"[GAP] VBM={gap_info['E_VBM_eV']:.3f} eV, CBM={gap_info['E_CBM_eV']:.3f} eV, "
          f"gap={gap_info['gap_eV']:.3f} eV ({'directo' if gap_info['direct'] else 'indirecto'})")
    print(f"[GAP] Bandas: {n_valence} valencia, {n_conduction} conducción")

    if gap_info['gap_eV'] < 0.1:  # Gap muy pequeño podría indicar problema
        print(f"[GAP] ¡ADVERTENCIA! Gap muy pequeño ({gap_info['gap_eV']:.3f} eV). "
              "Posibles problemas: estructura metálica o error en identificación de bandas.")

    return gap_info

def compute_bands_and_write(cell: gto.Cell, kmf, structure: Structure,
                            sigma_ha: float, basis: str,
                            ke_cutoff_Ry: float, kmesh: Tuple[int,int,int],
                            out_dir: Path, log: logging.Logger,
                            line_density: int = 50) -> Dict[str, Any]:
    kpts_frac, labels, seg_breaks = get_hs_kpath(structure, line_density=line_density)
    kdist = kpath_distances(cell, kpts_frac)
    b = cell.reciprocal_vectors(); kpts_cart = (kpts_frac @ b)
    bands = kmf.get_bands(kpts_cart)[0]
    gap_info = extract_gap_from_kmf(kmf)
    mu_ha = ev2ha(gap_info["mu_eV"]) if np.isfinite(gap_info["mu_eV"]) else 0.0
    occ_path = 1.0 / (1.0 + np.exp(((bands - mu_ha)/max(1e-12, sigma_ha))))
    bands_csv = out_dir / "bands.csv"
    with open(bands_csv, "w") as f:
        f.write("k_dist,kx,ky,kz,band_index,energy_eV,occ,VBM_eV,CBM_eV,is_direct\n")
        for ik in range(bands.shape[0]):
            for ib in range(bands.shape[1]):
                e_ev = ha2ev(bands[ik,ib]); occ = float(occ_path[ik,ib])
                f.write(f"{kdist[ik]:.8f},{kpts_frac[ik,0]:.8f},{kpts_frac[ik,1]:.8f},{kpts_frac[ik,2]:.8f},"
                        f"{ib},{e_ev:.8f},{occ:.6f},{gap_info['E_VBM_eV']:.6f},{gap_info['E_CBM_eV']:.6f},{int(gap_info['direct'])}\n")
    fig, ax = plt.subplots(figsize=(7.5,5.0))
    e_ev_all = ha2ev(bands)
    for ib in range(e_ev_all.shape[1]): ax.plot(kdist, e_ev_all[:,ib], lw=1.0, alpha=0.9)
    for br in seg_breaks: ax.axvline(kdist[br], color="k", lw=0.8, alpha=0.5)
    xticks = [0.0] + [kdist[i] for i in seg_breaks]
    ax.set_xticks(xticks); ax.set_xticklabels([""]*len(xticks))  # labels de HighSymmKpath varían; evitamos inconsistencias
    ax.set_ylabel("E (eV)")
    ax.set_title(f"Band structure | basis={basis}, cutoff={ke_cutoff_Ry} Ry, k={fmt_tuple(kmesh)}, σ={sigma_ha} Ha")
    ax.grid(True, ls="--", alpha=0.3)
    if np.isfinite(gap_info["E_VBM_eV"]) and np.isfinite(gap_info["E_CBM_eV"]):
        ax.axhline(gap_info["E_VBM_eV"], color="C3", ls="--", lw=1.2, label="VBM")
        ax.axhline(gap_info["E_CBM_eV"], color="C2", ls="--", lw=1.2, label="CBM")
        ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "bands.png", dpi=180); plt.close(fig)
    return gap_info

def compute_dos(kmf, out_dir: Path, sigma_ev: float = 0.1, ngrid: int = 2048, log: logging.Logger = None):
    mo_e = getattr(kmf, "mo_energy", [])
    if not mo_e: return
    evals = np.hstack([ha2ev(e) for e in mo_e])
    e_min, e_max = np.min(evals)-3.0, np.max(evals)+3.0
    E = np.linspace(e_min, e_max, ngrid); dos = np.zeros_like(E); w = sigma_ev
    for e in evals: dos += np.exp(-0.5*((E - e)/w)**2) / (w*np.sqrt(2*np.pi))
    pd.DataFrame({"E_eV":E, "DOS":dos}).to_csv(out_dir / "dos.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5,4.0))
    ax.plot(E, dos, lw=1.2); ax.set_xlabel("E (eV)"); ax.set_ylabel("DOS (a.u.)")
    ax.set_title("DOS (Gauss σ={:.2f} eV)".format(sigma_ev)); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out_dir / "dos.png", dpi=160); plt.close(fig)

# ---------------- Slab ----------------

from pymatgen.core.surface import SlabGenerator

def build_slab_cell_from_bulk(a_A: float, x_ga: float, miller: str, vacuum_A: float,
                              basis: str, pseudo: str, ke_cutoff_Ry: float) -> gto.Cell:
    bulk = build_pmg_structure(a_A, x_ga)
    miller = miller.strip()
    if miller not in ("001","110"): raise ValueError("Solo miller 001 o 110.")
    h = tuple(int(c) for c in list(miller))
    slabgen = SlabGenerator(bulk, h, min_slab_size=12.0, min_vacuum_size=vacuum_A, center_slab=True, in_unit_planes=True)
    slab = slabgen.get_slabs()[0]; lattice = slab.lattice
    cell = gto.Cell(); cell.unit = "A"; cell.a = lattice.matrix
    atoms = [(site.specie.symbol, (site.x, site.y, site.z)) for site in slab.sites]
    cell.atom = atoms
    cell.basis = {el: basis for el in set(slab.symbol_set)}
    cell.pseudo = {el: pseudo for el in set(slab.symbol_set)}
    cell.ke_cutoff = ry2ha(ke_cutoff_Ry); cell.exp_to_discard = 0.1
    cell.build()
    return cell

def cube_from_mf_potential(cell: gto.Cell, kmf, cube_path: Path, nx=80, ny=80, nz=200):
    from pyscf.tools import cubegen
    dm = kmf.make_rdm1(); cubegen.mep(cell, str(cube_path), dm, nx=nx, ny=ny, nz=nz)

def cube_from_mf_density(cell: gto.Cell, kmf, cube_path: Path, nx=80, ny=80, nz=200):
    from pyscf.tools import cubegen
    dm = kmf.make_rdm1(); cubegen.density(cell, str(cube_path), dm, nx=nx, ny=ny, nz=nz)

def planar_average_from_cube(cube_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with open(cube_file, "r") as f: lines = f.readlines()
    nat_line = lines[2].split(); natoms = int(nat_line[0])
    nx_line = lines[3].split(); ny_line = lines[4].split(); nz_line = lines[5].split()
    nx = abs(int(nx_line[0])); ny = abs(int(ny_line[0])); nz = abs(int(nz_line[0]))
    ax = np.array(list(map(float, nx_line[1:4]))); ay = np.array(list(map(float, ny_line[1:4]))); az = np.array(list(map(float, nz_line[1:4])))
    data_start = 6 + natoms; data_vals = []
    for line in lines[data_start:]: data_vals.extend([float(x) for x in line.split()])
    data_vals = np.array(data_vals); grid = data_vals.reshape((nz, ny, nx))
    Vz = grid.mean(axis=(1,2)); dz_bohr = np.linalg.norm(az); z_bohr = np.arange(nz) * dz_bohr
    z_A = z_bohr * BOHR_TO_ANG
    info = dict(nx=nx, ny=ny, nz=nz, ax=ax.tolist(), ay=ay.tolist(), az=az.tolist())
    return z_A, Vz, info

def slab_pipeline(a_A: float, x_ga: float, miller: str, vacuum_A: float,
                  basis: str, pseudo: str, sigma_ha: float, xc: str,
                  ke_cutoff_Ry: float, slab_kmesh: Tuple[int,int,int],
                  out_dir: Path, log: logging.Logger,
                  gpu: bool = False, conv_tol: float = 1e-8,
                  timeout_s:int = 600, policy:str="inline") -> Dict[str, Any]:
    cell = build_slab_cell_from_bulk(a_A, x_ga, miller, vacuum_A, basis, pseudo, ke_cutoff_Ry)
    ok, _, _ = run_with_policy((a_A, x_ga, basis, pseudo, ke_cutoff_Ry, slab_kmesh, xc, sigma_ha, conv_tol, gpu),
                               timeout_s, policy)
    kmf, _ = run_scf(cell, slab_kmesh, xc, sigma_ha, log, gpu=gpu, conv_tol=conv_tol)
    if not ok: log.warning("[slab] SCF timeout; Evac puede no ser confiable.")
    pot_cube = out_dir / "slab_potential.cube"; rho_cube = out_dir / "slab_density.cube"
    try:
        cube_from_mf_potential(cell, kmf, pot_cube)
        cube_from_mf_density(cell, kmf, rho_cube)
        z_A, Vz, _ = planar_average_from_cube(pot_cube)
        z_A_rho, rhoz, _ = planar_average_from_cube(rho_cube)
        rhoz = np.array(rhoz); thr = max(1e-6, 1e-3 * np.max(rhoz)); vac_mask = rhoz < thr
        if not np.any(vac_mask): vac_mask = np.ones_like(rhoz, dtype=bool)
        Evac = float(np.mean(np.array(Vz)[vac_mask]))
        fig, ax1 = plt.subplots(figsize=(7.5,4.5))
        ax1.plot(z_A, Vz, lw=1.1, label="⟨V⟩(z)")
        ax1.axhline(Evac, color="C3", ls="--", label=f"E_vac ≈ {Evac:.3f} Ha")
        ax1.set_xlabel("z (Å)"); ax1.set_ylabel("⟨V⟩ (Ha)"); ax1.grid(True, ls="--", alpha=0.4)
        ax12 = ax1.twinx(); ax12.plot(z_A_rho, rhoz, lw=0.8, alpha=0.5, color="C2", label="⟨ρ⟩(z)")
        ax12.set_ylabel("⟨ρ⟩ (a.u.)")
        ax1.set_title(f"Perfil planar | slab {miller}, vac={vacuum_A} Å, k={fmt_tuple(slab_kmesh)}")
        fig.tight_layout(); fig.savefig(out_dir / "potential_slab.png", dpi=180); plt.close(fig)
        return dict(Evac_Ha=Evac, success=True)
    except Exception as e:
        log.warning(f"[slab] No se pudo generar/analizar CUBE: {e}")
        return dict(Evac_Ha=np.nan, success=False)

# ============================ Report/metadata ============================

def write_simple_report(out_root: Path, info: Dict[str, Any]):
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Preconvergencia GaAs</title>
<style>body{{font-family:Arial, sans-serif;max-width:920px;margin:20px auto}}img{{max-width:100%}}</style></head>
<body>
<h1>Preconvergencia GaAs (PySCF)</h1>
<pre>{json.dumps(info, indent=2)}</pre>
<h2>Convergencia vs cutoff</h2><img src="cutoff/E_vs_cutoff.png"/>
<h2>Convergencia vs k-mesh</h2><img src="kmesh/E_vs_kmesh.png"/>
<h2>Ajuste E(a)</h2><img src="lattice/E_vs_a_fit.png"/>
<h2>Band structure</h2><img src="bands/bands.png"/>
{"<h2>Slab: potencial</h2><img src='slab/potential_slab.png'/>" if (out_root/'slab'/'potential_slab.png').exists() else ""}
</body></html>"""
    with open(out_root / "preconv_report.html", "w", encoding="utf-8") as f:
        f.write(html)

def write_metadata(out_root: Path, args: argparse.Namespace):
    meta = dict(
        timestamp = datetime.now().isoformat(),
        pyscf_version = getattr(sys.modules.get("pyscf"), "__version__", "unknown"),
        numpy_version = np.__version__,
        pandas_version = pd.__version__,
        matplotlib_version = matplotlib.__version__,
        n_cpus = mp.cpu_count(),
        args = vars(args),
        seed = args.seed
    )
    with open(out_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ============================ Main ============================

def validate_gaas_parameters(a_A: float, x_ga: float) -> Dict[str, Any]:
    """Valida parámetros estructurales de GaAs y retorna diagnóstico."""
    issues = []

    # Parámetro de red típico para GaAs: ~5.65 Å
    if not (5.0 <= a_A <= 6.5):
        issues.append(f"Parámetro de red a={a_A:.3f} Å fuera del rango típico para GaAs (5.0-6.5 Å)")

    # Posición fraccionaria típica para zincblende: 0.25
    if not (0.2 <= x_ga <= 0.3):
        issues.append(f"Posición x_ga={x_ga:.3f} fuera del rango típico para zincblende (0.2-0.3)")

    # Verificar que los átomos no estén demasiado cerca
    a1 = np.array([a_A, 0.0, 0.0])
    a2 = np.array([0.0, a_A, 0.0])
    a3 = np.array([0.0, 0.0, a_A])
    r_ga_frac = np.array([x_ga, x_ga, x_ga])
    r_ga_cart = r_ga_frac @ np.vstack([a1, a2, a3])
    distance = np.linalg.norm(r_ga_cart)

    if distance < 1.0:  # Umbral mínimo en Å
        issues.append(f"Distancia Ga-As={distance:.3f} Å demasiado pequeña (< 1.0 Å)")

    # Estimación del volumen de la celda
    volume = a_A**3 / 4  # Para FCC
    if not (40 <= volume <= 60):
        issues.append(f"Volumen de celda={volume:.1f} Å³ fuera del rango típico (40-60 Å³)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "distance_Ga_As": distance,
        "cell_volume": volume,
        "expected_gap_range": "1.4-1.5 eV (experimental)"
    }

def main():
    parser = argparse.ArgumentParser(description="Preconvergencia DFT/PBC para GaAs (PySCF) - Versión corregida.")
    parser.add_argument("--sigma_ha", type=float, default=0.01, help="Smearing Fermi–Dirac (Ha).")
    parser.add_argument("--cutoff_list", type=str, default=",".join(map(str, DEFAULT_CUTOFF_LIST_RY)), help="cutoffs (Ry, coma).")
    parser.add_argument("--k_list", type=str, default=",".join([fmt_tuple(k) for k in DEFAULT_K_LIST]), help="k-mesh list (e.g., 2x2x2,4x4x4).")
    parser.add_argument("--a0", type=float, default=5.653, help="a inicial (Å).")
    parser.add_argument("--da", type=float, default=0.03, help="paso de escaneo de a (Å).")
    parser.add_argument("--npoints_side", type=int, default=6, help="puntos por lado para E(a).")
    parser.add_argument("--basis_list", type=str, default=None, help="lista de bases GTH (coma).")
    parser.add_argument("--basis_sweep", type=str, default="off", help="on/off barrido de bases GTH.")
    parser.add_argument("--pseudo", type=str, default="gth-pbe", help="pseudo GTH.")
    parser.add_argument("--xc", type=str, default="PBE", help="funcional.")
    parser.add_argument("--x_ga", type=float, default=0.25, help="Ga en (x,x,x).")
    parser.add_argument("--dos", type=str, default="off", help="on/off DOS.")
    parser.add_argument("--fast", action="store_true", help="listas reducidas, ≥3 pts por lado.")
    parser.add_argument("--nprocs", type=int, default=max(1, mp.cpu_count()//2),
                        help="procesos (no usado en inline). En HPC usar OMP_NUM_THREADS.")
    parser.add_argument("--seed", type=int, default=12345, help="semilla RNG.")
    parser.add_argument("--plot_show", type=str, default="off", help="on/off mostrar plots.")
    parser.add_argument("--make_report", type=str, default="off", help="on/off HTML.")
    parser.add_argument("--gpu", type=str, default="off", help="on/off GPU.")
    parser.add_argument("--slab", type=str, default="off", help="on/off slab.")
    parser.add_argument("--slab_miller", type=str, default="001", help="miller slab (001/110).")
    parser.add_argument("--vacuum_A", type=float, default=12.0, help="vacío (Å).")
    parser.add_argument("--slab_kmesh", type=str, default="6x6x1", help="kmesh slab.")
    parser.add_argument("--timeout_s", type=int, default=300, help="timeout por punto (seg).")
    parser.add_argument("--timeout_policy", type=str, default="inline", choices=["inline","subproc"],
                        help="inline (seguro) o subproc (usa procesos).")
    parser.add_argument("--resume", type=str, default="off",
                        help="on/off reanudar desde checkpoint previo.")
    parser.add_argument("--checkpoint_interval", type=int, default=300,
                        help="Intervalo de guardado de checkpoints (segundos).")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Validación de parámetros estructurales
    validation = validate_gaas_parameters(args.a0, args.x_ga)
    print("=== VALIDACIÓN DE PARÁMETROS ESTRUCTURALES ===")
    print(f"Parámetro de red: {args.a0:.4f} Å")
    print(f"Posición Ga (x,x,x): {args.x_ga}")
    print(f"Distancia Ga-As: {validation['distance_Ga_As']:.4f} Å")
    print(f"Volumen de celda: {validation['cell_volume']:.2f} Å³")

    if not validation['valid']:
        print("¡PROBLEMAS DETECTADOS!")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print(f"\nRango de gap esperado: {validation['expected_gap_range']}")
        print("Considere ajustar los parámetros antes de continuar.\n")
    else:
        print("✓ Parámetros estructurales válidos")

    # Validación de reproducibilidad si existe validador
    try:
        from reproducibility_validator import ReproducibilityValidator
        validator = ReproducibilityValidator(out_root)
        env_fingerprint = validator.get_environment_fingerprint()
        print("
=== HUELLA DIGITAL DEL ENTORNO ===")
        print(f"Fingerprint: {env_fingerprint['fingerprint']}")
        print(f"Python: {env_fingerprint['python_version'][:50]}...")
        print(f"PySCF: {env_fingerprint['pyscf_version']}")
        print(f"OMP_NUM_THREADS: {env_fingerprint['environment_variables'].get('OMP_NUM_THREADS', 'NOT_SET')}")
        print("=====================================")
    except ImportError:
        print("Nota: Validador de reproducibilidad no disponible")

    out_root = Path("preconvergencia_out")
    subdirs = ensure_dirs(out_root)
    log = setup_logging(out_root)
    write_metadata(out_root, args)

    # Cargar checkpoint si existe
    log.info("🔍 Verificando checkpoints previos...")
    checkpoint = load_latest_checkpoint(out_root)
    resume_from_stage = checkpoint.get("stage") if checkpoint else None

    if resume_from_stage:
        log.info(f"📁 Checkpoint encontrado: {resume_from_stage}")
        log.info("💡 Puedes usar --resume para continuar desde este punto")
    else:
        log.info("📁 No se encontraron checkpoints previos")

    nprocs = max(1, int(args.nprocs))
    # Configuración inteligente de paralelización para HPC
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", max(1, nprocs)))
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    pyscf_lib.num_threads(omp_threads)

    cutoff_list = parse_cutoff_list(args.cutoff_list)
    k_list = parse_k_list(args.k_list)
    basis_sweep = str2bool(args.basis_sweep)
    use_gpu = str2bool(args.gpu)
    do_dos = str2bool(args.dos)
    show_plots = str2bool(args.plot_show)
    make_report = str2bool(args.make_report)
    do_slab = str2bool(args.slab)
    slab_kmesh = parse_k_list(args.slab_kmesh)[0]
    timeout_s = int(args.timeout_s)
    policy = args.timeout_policy
    enable_resume = str2bool(args.resume)
    checkpoint_interval = int(args.checkpoint_interval)

    if args.fast:
        log.info("[FAST] Activado. Reduciendo listas y puntos.")
        if len(cutoff_list) > 4:
            cutoff_list = sorted(set([cutoff_list[0], cutoff_list[len(cutoff_list)//2], cutoff_list[-2], cutoff_list[-1]]))
        if len(k_list) > 5:
            k_list = [(2,2,2),(4,4,4),(6,6,6),(8,8,8),(12,12,12)]
        args.npoints_side = max(3, min(args.npoints_side, 4))

    if basis_sweep:
        cand = UNIVERSAL_GTH_BASES
        if args.basis_list:
            cand = [b.strip() for b in args.basis_list.split(",") if b.strip()]
        bases = filter_available_gth_bases(cand, log)
        if not bases:
            bases = ["gth-dzvp"]; log.warning("[bases] Ninguna candidata; usando gth-dzvp.")
    else:
        # Usar bases más estándar disponibles en PySCF
        default_basis = "def2-svp"  # Base más estándar disponible
        bases = [args.basis_list.strip()] if args.basis_list else [default_basis]
        # No filtrar por GTH ya que usaremos bases estándar
        log.info(f"[bases] Usando bases estándar: {bases}")

    log.info(f"[bases] Seleccionadas: {bases}")

    gap_summary_rows = []; anderson_rows = []

    for basis in bases:
        log.info(f"================ Base: {basis} ================")

        # Guardar checkpoint antes de cutoff
        save_checkpoint(out_root, "pre_cutoff", {"basis": basis, "cutoff_list": cutoff_list}, log)

        # 1) cutoff
        df_cutoff = scan_cutoff(a_A=args.a0, x_ga=args.x_ga, basis=basis, pseudo=args.pseudo,
                                sigma_ha=args.sigma_ha, xc=args.xc,
                                cutoff_list_Ry=cutoff_list, kmesh_fixed=(6,6,6),
                                out_dir=subdirs["cutoff"], log=log,
                                nprocs=nprocs, gpu=use_gpu, timeout_s=timeout_s, policy=policy)
        cutoff_star = choose_cutoff(df_cutoff, thr_Ha=1e-4) if df_cutoff is not None else cutoff_list[-1]
        if cutoff_star is None:
            cutoff_star = 100.0  # Valor por defecto si no se puede determinar
            log.warning(f"[cutoff*] No se pudo determinar cutoff óptimo, usando {cutoff_star} Ry por defecto")
        log.info(f"[cutoff*] {cutoff_star} Ry")

        # Guardar checkpoint después de cutoff
        save_checkpoint(out_root, "post_cutoff", {"basis": basis, "cutoff_star": cutoff_star}, log)

        # 2) k-mesh
        save_checkpoint(out_root, "pre_kmesh", {"basis": basis, "cutoff_star": cutoff_star}, log)
        df_k = scan_kmesh(a_A=args.a0, x_ga=args.x_ga, basis=basis, pseudo=args.pseudo,
                          sigma_ha=args.sigma_ha, xc=args.xc,
                          ke_cutoff_Ry=cutoff_star, k_list=k_list,
                          out_dir=subdirs["kmesh"], log=log,
                          nprocs=1, gpu=use_gpu, timeout_s=timeout_s, policy=policy)
        kmesh_star = choose_kmesh(df_k, thr_Ha=1e-5) if df_k is not None else k_list[-1]
        if kmesh_star is None:
            kmesh_star = (6, 6, 6)  # Valor por defecto
            log.warning(f"[k*] No se pudo determinar kmesh óptimo, usando {fmt_tuple(kmesh_star)} por defecto")
        log.info(f"[k*] {fmt_tuple(kmesh_star)}")

        save_checkpoint(out_root, "post_kmesh", {"basis": basis, "kmesh_star": kmesh_star}, log)

        # 3) E(a) + ajuste avanzado
        save_checkpoint(out_root, "pre_lattice", {"basis": basis, "cutoff_star": cutoff_star, "kmesh_star": kmesh_star}, log)
        df_lat, fit = advanced_lattice_optimization(a0=args.a0, da=args.da, npoints_side=args.npoints_side,
                                                    x_ga=args.x_ga, basis=basis, pseudo=args.pseudo,
                                                    sigma_ha=args.sigma_ha, xc=args.xc,
                                                    ke_cutoff_Ry=cutoff_star, kmesh=kmesh_star,
                                                    out_dir=subdirs["lattice"], log=log,
                                                    nprocs=nprocs, gpu=use_gpu, timeout_s=timeout_s, policy=policy,
                                                    n_random_restarts=3, enable_multi_start=True)
        a_opt = float(fit["a_opt"]) if fit else args.a0
        E_min = float(fit["E_min"]) if fit else np.nan
        log.info(f"[a*] a_opt ≈ {a_opt:.6f} Å (R²={fit['R2']:.6f if fit else np.nan})")

        save_checkpoint(out_root, "post_lattice", {"basis": basis, "a_opt": a_opt, "fit_info": fit}, log)

        # 4) Bandas (+DOS opcional)
        save_checkpoint(out_root, "pre_bands", {"basis": basis, "a_opt": a_opt}, log)
        cell_final = build_gaas_cell(a_opt, args.x_ga, basis, args.pseudo, cutoff_star)
        kmf_final, ok = run_scf(cell_final, kmesh_star, args.xc, args.sigma_ha, log, gpu=use_gpu)
        if not ok: log.warning("[final] SCF no convergió; resultados pueden contener NaN.")
        gap_info = extract_gap_from_kmf(kmf_final)
        gap_info["basis"] = basis; gap_info["a_opt_Ang"] = a_opt; gap_info["E_min_Ha"] = E_min
        struct = build_pmg_structure(a_opt, args.x_ga)
        _ = compute_bands_and_write(cell_final, kmf_final, struct, args.sigma_ha, basis,
                                    cutoff_star, kmesh_star, subdirs["bands"], log, line_density=50)
        if do_dos:
            compute_dos(kmf_final, subdirs["dos"], sigma_ev=0.15, log=log)

        save_checkpoint(out_root, "post_bands", {"basis": basis, "gap_info": gap_info}, log)

        gap_summary_rows.append(dict(
            basis=basis, a_opt_Ang=a_opt, E_min_Ha=E_min,
            gap_eV=gap_info["gap_eV"],
            direct_indirect="direct" if gap_info.get("direct", False) else "indirect",
            k_VBM=str(gap_info.get("k_VBM")), k_CBM=str(gap_info.get("k_CBM"))
        ))
        pd.DataFrame(gap_summary_rows).to_csv(subdirs["bands"] / "gap_summary.csv", index=False)

        # 5) Slab (opcional)
        if do_slab:
            save_checkpoint(out_root, "pre_slab", {"basis": basis, "a_opt": a_opt}, log)
            slab_res = slab_pipeline(a_opt, args.x_ga, args.slab_miller, args.vacuum_A,
                                     basis, args.pseudo, args.sigma_ha, args.xc,
                                     cutoff_star, slab_kmesh, subdirs["slab"], log, gpu=use_gpu,
                                     timeout_s=max(timeout_s, 600), policy=policy)
            midgap_eV = gap_info.get("mu_eV", np.nan)
            Evac_eV = ha2ev(slab_res["Evac_Ha"]) if np.isfinite(slab_res.get("Evac_Ha", np.nan)) else np.nan
            chi = (Evac_eV - gap_info.get("E_CBM_eV", np.nan)) if np.isfinite(Evac_eV) else np.nan
            phi_mid = (Evac_eV - midgap_eV) if np.isfinite(Evac_eV) else np.nan
            pd.DataFrame([dict(
                basis=basis, a_opt_Ang=a_opt, midgap_EF_eV=midgap_eV,
                Evac_eV=Evac_eV, chi_eV=chi, phi_midgap_eV=phi_mid
            )]).to_csv(subdirs["slab"] / "anderson_alignment.csv", index=False)

            save_checkpoint(out_root, "post_slab", {"basis": basis, "slab_results": slab_res}, log)

    # Guardar checkpoint final
    save_checkpoint(out_root, "completed", {"status": "completed", "bases_processed": bases}, log)

    if make_report:
        info = dict(bases=bases, cutoff_list_Ry=cutoff_list,
                    k_list=[fmt_tuple(k) for k in k_list], sigma_Ha=args.sigma_ha)
        write_simple_report(out_root, info)

    # Ejecutar validación de reproducibilidad al final
    try:
        from reproducibility_validator import ReproducibilityValidator
        validator = ReproducibilityValidator(out_root)
        validation_report = validator.save_validation_report()
        log.info(f"[VALIDATION] Reporte de reproducibilidad guardado: {validation_report}")

        # Mostrar resumen de validación
        with open(validation_report, 'r') as f:
            report_data = json.load(f)

        summary = report_data.get("summary", {})
        log.info(f"[VALIDATION] Estado general: {summary.get('overall_status', 'UNKNOWN')}")

        if summary.get('errors'):
            log.error("[VALIDATION] Errores encontrados:")
            for error in summary['errors']:
                log.error(f"  - {error}")

        if summary.get('warnings'):
            log.warning("[VALIDATION] Advertencias:")
            for warning in summary['warnings']:
                log.warning(f"  - {warning}")

    except ImportError:
        log.info("[VALIDATION] Validador de reproducibilidad no disponible")
    except Exception as e:
        log.warning(f"[VALIDATION] Error en validación de reproducibilidad: {e}")

    if show_plots: plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error fatal:", e)
        traceback.print_exc()
        sys.exit(1)
