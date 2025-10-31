#!/usr/bin/env python3
"""
Validador de reproducibilidad para cálculos DFT de GaAs.
Asegura consistencia entre entornos locales y HPC.
"""

import os
import sys
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess

import numpy as np
import pandas as pd


class ReproducibilityValidator:
    """Validador de reproducibilidad para cálculos DFT."""

    def __init__(self, results_dir: Path = Path("preconvergencia_out")):
        self.results_dir = results_dir
        self.validation_log = []

    def log_validation(self, message: str, level: str = "INFO"):
        """Registrar mensaje de validación."""
        timestamp = datetime.now().isoformat()
        self.validation_log.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })
        print(f"[{level}] {message}")

    def get_environment_fingerprint(self) -> Dict[str, Any]:
        """Generar huella digital del entorno de ejecución."""
        env_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "environment_variables": {},
        }

        # Variables de entorno relevantes
        relevant_vars = [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PYSCF_MAX_MEMORY", "SLURM_JOB_ID", "SLURM_JOB_PARTITION",
            "SLURM_NNODES", "SLURM_CPUS_PER_TASK"
        ]

        for var in relevant_vars:
            env_info["environment_variables"][var] = os.environ.get(var, "NOT_SET")

        # Información de bibliotecas
        try:
            import pyscf
            env_info["pyscf_version"] = getattr(pyscf, "__version__", "unknown")
        except:
            env_info["pyscf_version"] = "not_available"

        try:
            import numpy
            env_info["numpy_version"] = numpy.__version__
        except:
            env_info["numpy_version"] = "not_available"

        try:
            import scipy
            env_info["scipy_version"] = scipy.__version__
        except:
            env_info["scipy_version"] = "not_available"

        # Generar hash del entorno
        env_str = json.dumps(env_info, sort_keys=True)
        env_info["fingerprint"] = hashlib.sha256(env_str.encode()).hexdigest()[:16]

        return env_info

    def validate_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """Validar integridad de archivos críticos."""
        if not file_path.exists():
            return {"status": "MISSING", "path": str(file_path)}

        try:
            # Calcular hash del archivo
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Obtener metadatos
            stat = file_path.stat()
            file_info = {
                "status": "OK",
                "path": str(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "hash": file_hash[:16]  # Primeros 16 caracteres
            }

            return file_info

        except Exception as e:
            return {
                "status": "ERROR",
                "path": str(file_path),
                "error": str(e)
            }

    def validate_convergence_results(self) -> Dict[str, Any]:
        """Validar consistencia de resultados de convergencia."""
        validation_results = {
            "cutoff_convergence": self._validate_cutoff_convergence(),
            "kmesh_convergence": self._validate_kmesh_convergence(),
            "lattice_optimization": self._validate_lattice_optimization(),
            "band_structure": self._validate_band_structure()
        }

        return validation_results

    def _validate_cutoff_convergence(self) -> Dict[str, Any]:
        """Validar convergencia de cutoff."""
        cutoff_csv = self.results_dir / "cutoff" / "cutoff.csv"
        if not cutoff_csv.exists():
            return {"status": "NO_DATA"}

        try:
            df = pd.read_csv(cutoff_csv)
            if df.empty:
                return {"status": "EMPTY"}

            # Verificar que las energías decrecen con cutoff
            energies = df["E_tot_Ha"].dropna().values
            if len(energies) < 2:
                return {"status": "INSUFFICIENT_DATA"}

            # Calcular diferencias de energía
            energy_diffs = np.diff(energies)
            convergence_rate = np.abs(energy_diffs[-1]) if len(energy_diffs) > 0 else float('inf')

            # Verificar criterio de convergencia (1 meV = 0.001 Ha)
            converged = convergence_rate < 0.001

            return {
                "status": "OK" if converged else "NOT_CONVERGED",
                "n_points": len(energies),
                "final_energy_diff_ha": convergence_rate,
                "convergence_threshold_ha": 0.001,
                "energies_range": [float(energies.min()), float(energies.max())]
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _validate_kmesh_convergence(self) -> Dict[str, Any]:
        """Validar convergencia de k-mesh."""
        kmesh_csv = self.results_dir / "kmesh" / "kmesh.csv"
        if not kmesh_csv.exists():
            return {"status": "NO_DATA"}

        try:
            df = pd.read_csv(kmesh_csv)
            if df.empty:
                return {"status": "EMPTY"}

            # Verificar que las energías convergen con k-mesh
            energies = df["E_tot_Ha"].dropna().values
            if len(energies) < 2:
                return {"status": "INSUFFICIENT_DATA"}

            # Calcular diferencias
            energy_diffs = np.abs(np.diff(energies))
            convergence_rate = energy_diffs[-1] if len(energy_diffs) > 0 else float('inf')

            converged = convergence_rate < 0.0001  # 0.1 meV

            return {
                "status": "OK" if converged else "NOT_CONVERGED",
                "n_points": len(energies),
                "final_energy_diff_ha": convergence_rate,
                "convergence_threshold_ha": 0.0001,
                "kmesh_range": [df["N_kpts"].min(), df["N_kpts"].max()]
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _validate_lattice_optimization(self) -> Dict[str, Any]:
        """Validar optimización de parámetro de red."""
        lattice_csv = self.results_dir / "lattice" / "lattice_optimization.csv"
        if not lattice_csv.exists():
            return {"status": "NO_DATA"}

        try:
            df = pd.read_csv(lattice_csv)
            if df.empty:
                return {"status": "EMPTY"}

            # Verificar ajuste cuadrático
            valid_points = df.dropna(subset=["E_tot_Ha"])
            if len(valid_points) < 3:
                return {"status": "INSUFFICIENT_DATA"}

            # Verificar que el parámetro de red esté en rango físico
            a_opt = valid_points["a_Ang"].mean()  # Aproximación
            if not (5.5 <= a_opt <= 5.8):
                return {
                    "status": "OUT_OF_RANGE",
                    "a_opt": a_opt,
                    "expected_range": [5.5, 5.8]
                }

            return {
                "status": "OK",
                "n_points": len(valid_points),
                "a_range": [valid_points["a_Ang"].min(), valid_points["a_Ang"].max()],
                "energy_range": [valid_points["E_tot_Ha"].min(), valid_points["E_tot_Ha"].max()]
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _validate_band_structure(self) -> Dict[str, Any]:
        """Validar estructura de bandas."""
        bands_csv = self.results_dir / "bands" / "bands.csv"
        if not bands_csv.exists():
            return {"status": "NO_DATA"}

        try:
            df = pd.read_csv(bands_csv)
            if df.empty:
                return {"status": "EMPTY"}

            # Verificar gap
            gap_ev = df["gap_eV"].dropna().iloc[0] if "gap_eV" in df.columns else None
            if gap_ev is None:
                return {"status": "NO_GAP_DATA"}

            # Gap esperado para GaAs: ~1.4 eV
            if not (0.5 <= gap_ev <= 2.5):
                return {
                    "status": "GAP_OUT_OF_RANGE",
                    "gap_ev": gap_ev,
                    "expected_range": [0.5, 2.5]
                }

            return {
                "status": "OK",
                "gap_ev": gap_ev,
                "is_direct": df["is_direct"].iloc[0] if "is_direct" in df.columns else None
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def validate_software_versions(self) -> Dict[str, Any]:
        """Validar versiones de software."""
        versions = {}

        # PySCF
        try:
            import pyscf
            versions["pyscf"] = getattr(pyscf, "__version__", "unknown")
        except:
            versions["pyscf"] = "not_installed"

        # NumPy
        try:
            import numpy
            versions["numpy"] = numpy.__version__
        except:
            versions["numpy"] = "not_installed"

        # SciPy
        try:
            import scipy
            versions["scipy"] = scipy.__version__
        except:
            versions["scipy"] = "not_installed"

        # Pandas
        try:
            import pandas
            versions["pandas"] = pandas.__version__
        except:
            versions["pandas"] = "not_installed"

        # PyMatGen
        try:
            import pymatgen
            versions["pymatgen"] = getattr(pymatgen, "__version__", "unknown")
        except:
            versions["pymatgen"] = "not_installed"

        return versions

    def run_full_validation(self) -> Dict[str, Any]:
        """Ejecutar validación completa."""
        self.log_validation("Iniciando validación de reproducibilidad")

        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "environment_fingerprint": self.get_environment_fingerprint(),
            "software_versions": self.validate_software_versions(),
            "file_integrity": {},
            "convergence_validation": self.validate_convergence_results(),
            "validation_log": self.validation_log
        }

        # Validar archivos críticos
        critical_files = [
            "run_metadata.json",
            "cutoff/cutoff.csv",
            "kmesh/kmesh.csv",
            "lattice/lattice_optimization.csv",
            "bands/bands.csv"
        ]

        for file_path in critical_files:
            full_path = self.results_dir / file_path
            validation_report["file_integrity"][file_path] = self.validate_file_integrity(full_path)

        # Resumen de validación
        validation_report["summary"] = self._generate_validation_summary(validation_report)

        return validation_report

    def _generate_validation_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generar resumen de validación."""
        summary = {
            "overall_status": "PASS",
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        # Verificar estado de convergencia
        conv_results = report["convergence_validation"]

        for test_name, result in conv_results.items():
            if result.get("status") not in ["OK", "NO_DATA"]:
                summary["warnings"].append(f"{test_name}: {result.get('status')}")

        # Verificar integridad de archivos
        for file_path, file_info in report["file_integrity"].items():
            if file_info.get("status") != "OK":
                summary["errors"].append(f"Archivo {file_path}: {file_info.get('status')}")

        # Generar recomendaciones
        if summary["errors"]:
            summary["overall_status"] = "FAIL"
            summary["recommendations"].append("Corregir errores de archivos antes de continuar")
        elif summary["warnings"]:
            summary["overall_status"] = "WARN"
            summary["recommendations"].append("Revisar advertencias de convergencia")

        return summary

    def save_validation_report(self, output_file: Path = None):
        """Guardar reporte de validación."""
        if output_file is None:
            output_file = self.results_dir / "reproducibility_validation.json"

        report = self.run_full_validation()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.log_validation(f"Reporte de validación guardado: {output_file}")
        return output_file


def main():
    """Función principal para línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(description="Validador de reproducibilidad para cálculos DFT de GaAs")
    parser.add_argument("--results-dir", type=Path, default=Path("preconvergencia_out"),
                        help="Directorio de resultados")
    parser.add_argument("--output-file", type=Path,
                        help="Archivo de salida para el reporte")

    args = parser.parse_args()

    validator = ReproducibilityValidator(args.results_dir)
    output_file = validator.save_validation_report(args.output_file)

    print(f"\nValidación completada. Reporte guardado en: {output_file}")

    # Mostrar resumen
    with open(output_file, 'r') as f:
        report = json.load(f)

    summary = report["summary"]
    print(f"\nEstado general: {summary['overall_status']}")

    if summary['errors']:
        print("Errores encontrados:")
        for error in summary['errors']:
            print(f"  - {error}")

    if summary['warnings']:
        print("Advertencias:")
        for warning in summary['warnings']:
            print(f"  - {warning}")

    if summary['recommendations']:
        print("Recomendaciones:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()