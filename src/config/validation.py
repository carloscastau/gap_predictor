# src/config/validation.py
"""Validación de configuración y parámetros."""

from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path


class ConfigValidator:
    """Validador de configuración del sistema."""

    @staticmethod
    def validate_lattice_parameters(a: float, x_ga: float) -> Dict[str, Any]:
        """Valida parámetros estructurales de GaAs."""
        issues = []
        warnings = []

        # Validar parámetro de red
        if not (5.0 <= a <= 6.5):
            issues.append(f"Parámetro de red a={a:.3f} Å fuera del rango físico típico (5.0-6.5 Å)")

        # Validar posición fraccionaria
        if not (0.2 <= x_ga <= 0.3):
            issues.append(f"Posición x_ga={x_ga:.3f} fuera del rango típico para zincblende (0.2-0.3)")

        # Calcular distancia Ga-As
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
        a3 = np.array([0.0, 0.0, a])
        r_ga_frac = np.array([x_ga, x_ga, x_ga])
        r_ga_cart = r_ga_frac @ np.vstack([a1, a2, a3])
        distance_ga_as = np.linalg.norm(r_ga_cart)

        if distance_ga_as < 1.0:
            issues.append(f"Distancia Ga-As={distance_ga_as:.3f} Å demasiado pequeña (< 1.0 Å)")
        elif distance_ga_as > 3.0:
            warnings.append(f"Distancia Ga-As={distance_ga_as:.3f} Å inusualmente grande (> 3.0 Å)")

        # Calcular volumen de celda
        volume = a ** 3 / 4  # Para estructura FCC
        if not (40 <= volume <= 60):
            warnings.append(f"Volumen de celda={volume:.1f} Å³ fuera del rango típico (40-60 Å³)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'distance_ga_as': distance_ga_as,
            'cell_volume': volume,
            'expected_gap_range': '1.4-1.5 eV (experimental)'
        }

    @staticmethod
    def validate_cutoff_list(cutoff_list: List[float]) -> Dict[str, Any]:
        """Valida lista de cutoffs."""
        issues = []
        warnings = []

        if not cutoff_list:
            issues.append("Lista de cutoffs vacía")
            return {'valid': False, 'issues': issues, 'warnings': warnings}

        if len(cutoff_list) < 2:
            issues.append("Se requieren al menos 2 valores de cutoff para análisis de convergencia")

        # Verificar orden ascendente
        if not all(cutoff_list[i] <= cutoff_list[i+1] for i in range(len(cutoff_list)-1)):
            issues.append("Lista de cutoffs debe estar ordenada ascendentemente")

        # Verificar valores razonables (en Ry)
        for cutoff in cutoff_list:
            if cutoff < 20:
                issues.append(f"Cutoff {cutoff} Ry demasiado pequeño (< 20 Ry)")
            elif cutoff > 500:
                warnings.append(f"Cutoff {cutoff} Ry muy grande (> 500 Ry)")

        # Verificar espaciado
        if len(cutoff_list) > 1:
            spacings = np.diff(cutoff_list)
            avg_spacing = np.mean(spacings)
            if avg_spacing < 10:
                warnings.append(f"Espaciado promedio entre cutoffs ({avg_spacing:.1f} Ry) muy pequeño")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'range': f"{min(cutoff_list)}-{max(cutoff_list)} Ry" if cutoff_list else None
        }

    @staticmethod
    def validate_kmesh_list(kmesh_list: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Valida lista de mallas k-point."""
        issues = []
        warnings = []

        if not kmesh_list:
            issues.append("Lista de k-mesh vacía")
            return {'valid': False, 'issues': issues, 'warnings': warnings}

        if len(kmesh_list) < 2:
            issues.append("Se requieren al menos 2 mallas k-point para análisis de convergencia")

        # Verificar que sean tuplas de 3 enteros positivos
        for kmesh in kmesh_list:
            if not isinstance(kmesh, (tuple, list)) or len(kmesh) != 3:
                issues.append(f"k-mesh {kmesh} debe ser tupla/lista de 3 elementos")
                continue

            if not all(isinstance(k, int) and k > 0 for k in kmesh):
                issues.append(f"k-mesh {kmesh} debe contener enteros positivos")

        # Verificar orden por número total de k-points
        if len(kmesh_list) > 1:
            nkpts = [k[0] * k[1] * k[2] for k in kmesh_list]
            if not all(nkpts[i] <= nkpts[i+1] for i in range(len(nkpts)-1)):
                warnings.append("k-mesh no están ordenados por número total de k-points")

        # Verificar mallas muy grandes
        for kmesh in kmesh_list:
            nk_total = kmesh[0] * kmesh[1] * kmesh[2]
            if nk_total > 1000:
                warnings.append(f"k-mesh {kmesh} muy grande ({nk_total} k-points)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'max_nkpts': max((k[0] * k[1] * k[2] for k in kmesh_list)) if kmesh_list else 0
        }

    @staticmethod
    def validate_basis_availability(basis: str, pseudopotential: str) -> Dict[str, Any]:
        """Valida disponibilidad de basis set y pseudopotencial."""
        issues = []
        warnings = []

        # Verificar formato de basis
        if not basis or not isinstance(basis, str):
            issues.append("Basis set debe ser string no vacío")
            return {'valid': False, 'issues': issues, 'warnings': warnings}

        # Para bases GTH, verificar formato
        if 'gth' in basis.lower():
            valid_gth_bases = ['gth-szv', 'gth-dzv', 'gth-dzvp', 'gth-tzvp', 'gth-tzv2p', 'gth-qzv2p']
            if basis.lower() not in valid_gth_bases:
                warnings.append(f"Basis GTH '{basis}' no está en lista estándar de PySCF")

        # Verificar pseudopotencial
        if pseudopotential and 'gth' in pseudopotential.lower():
            if not pseudopotential.lower().startswith('gth-'):
                warnings.append(f"Pseudopotencial '{pseudopotential}' no sigue convención GTH")

        # Nota: La validación real requiere importar PySCF, se hace en runtime
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'requires_runtime_check': True
        }

    @staticmethod
    def validate_system_resources(memory_limit_gb: float, max_workers: int) -> Dict[str, Any]:
        """Valida recursos del sistema."""
        import psutil
        import multiprocessing

        issues = []
        warnings = []

        # Obtener recursos disponibles
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        available_cpus = multiprocessing.cpu_count()

        # Validar memoria
        if memory_limit_gb <= 0:
            issues.append("Límite de memoria debe ser positivo")
        elif memory_limit_gb > available_memory * 0.9:
            warnings.append(f"Límite de memoria ({memory_limit_gb} GB) > 90% de memoria disponible ({available_memory:.1f} GB)")

        # Validar CPUs
        if max_workers <= 0:
            issues.append("Número de workers debe ser positivo")
        elif max_workers > available_cpus:
            warnings.append(f"max_workers ({max_workers}) > CPUs disponibles ({available_cpus})")
        elif max_workers > available_cpus * 0.8:
            warnings.append(f"max_workers ({max_workers}) > 80% de CPUs disponibles ({available_cpus})")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'available_memory_gb': available_memory,
            'available_cpus': available_cpus,
            'recommended_max_workers': min(available_cpus, max(1, available_cpus // 2))
        }

    @staticmethod
    def validate_output_directory(output_dir: Path) -> Dict[str, Any]:
        """Valida directorio de salida."""
        issues = []
        warnings = []

        try:
            # Intentar crear directorio
            output_dir.mkdir(parents=True, exist_ok=True)

            # Verificar permisos de escritura
            test_file = output_dir / '.write_test'
            test_file.write_text('test')
            test_file.unlink()

        except Exception as e:
            issues.append(f"No se puede escribir en directorio de salida: {e}")

        # Verificar espacio disponible
        try:
            stat = os.statvfs(str(output_dir))
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            if available_gb < 1.0:
                warnings.append(f"Espacio disponible bajo: {available_gb:.1f} GB")
        except:
            pass  # No crítico

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'path': str(output_dir.resolve())
        }


def validate_full_config(config) -> Dict[str, Any]:
    """Valida configuración completa."""
    results = {}

    # Validar parámetros físicos
    results['lattice'] = ConfigValidator.validate_lattice_parameters(
        config.lattice_constant, config.x_ga
    )

    # Validar parámetros de convergencia
    results['cutoff'] = ConfigValidator.validate_cutoff_list(config.cutoff_list)
    results['kmesh'] = ConfigValidator.validate_kmesh_list(config.kmesh_list)

    # Validar basis y pseudopotencial
    results['basis'] = ConfigValidator.validate_basis_availability(
        config.basis_set, config.pseudopotential
    )

    # Validar recursos del sistema
    results['resources'] = ConfigValidator.validate_system_resources(
        config.memory_limit_gb, config.max_workers
    )

    # Validar directorio de salida
    results['output'] = ConfigValidator.validate_output_directory(config.output_dir)

    # Resumen general
    all_issues = []
    all_warnings = []

    for category, result in results.items():
        all_issues.extend(result.get('issues', []))
        all_warnings.extend(result.get('warnings', []))

    results['summary'] = {
        'valid': len(all_issues) == 0,
        'total_issues': len(all_issues),
        'total_warnings': len(all_warnings),
        'issues': all_issues,
        'warnings': all_warnings
    }

    return results