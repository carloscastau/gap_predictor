# src/config/hardware.py
"""Configuración y detección de hardware."""

import os
import psutil
import multiprocessing
from typing import Dict, Any, Optional
import subprocess


class HardwareDetector:
    """Detector de hardware del sistema."""

    @staticmethod
    def detect_cpu_info() -> Dict[str, Any]:
        """Detecta información de CPU."""
        cpu_info = {
            'count': multiprocessing.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'usage_percent': psutil.cpu_percent(interval=1)
        }

        # Intentar obtener más información del sistema
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Extraer modelo de CPU
                for line in cpuinfo.split('\n'):
                    if line.startswith('model name'):
                        cpu_info['model'] = line.split(':')[1].strip()
                        break
        except:
            cpu_info['model'] = 'Unknown'

        return cpu_info

    @staticmethod
    def detect_memory_info() -> Dict[str, Any]:
        """Detecta información de memoria."""
        mem = psutil.virtual_memory()

        return {
            'total_gb': mem.total / (1024 ** 3),
            'available_gb': mem.available / (1024 ** 3),
            'used_gb': mem.used / (1024 ** 3),
            'usage_percent': mem.percent,
            'swap_total_gb': psutil.swap_memory().total / (1024 ** 3),
            'swap_used_gb': psutil.swap_memory().used / (1024 ** 3)
        }

    @staticmethod
    def detect_gpu_info() -> Dict[str, Any]:
        """Detecta información de GPU."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'libraries': {}
        }

        # Verificar CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['count'] = torch.cuda.device_count()
                gpu_info['libraries']['pytorch'] = True

                devices = []
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    devices.append({
                        'id': i,
                        'name': device_props.name,
                        'memory_total_mb': device_props.total_memory / (1024 ** 2),
                        'compute_capability': f"{device_props.major}.{device_props.minor}"
                    })
                gpu_info['devices'] = devices
        except ImportError:
            gpu_info['libraries']['pytorch'] = False

        # Verificar gpu4pyscf
        try:
            import gpu4pyscf
            gpu_info['libraries']['gpu4pyscf'] = True
        except ImportError:
            gpu_info['libraries']['gpu4pyscf'] = False

        return gpu_info

    @staticmethod
    def detect_storage_info(path: str = ".") -> Dict[str, Any]:
        """Detecta información de almacenamiento."""
        try:
            stat = os.statvfs(path)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            used_gb = total_gb - available_gb

            return {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'used_gb': used_gb,
                'usage_percent': (used_gb / total_gb) * 100 if total_gb > 0 else 0
            }
        except Exception as e:
            return {
                'error': str(e),
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'usage_percent': 0
            }

    @staticmethod
    def detect_network_info() -> Dict[str, Any]:
        """Detecta información de red."""
        try:
            # Obtener interfaces de red
            net_interfaces = psutil.net_if_addrs()
            interfaces_info = {}

            for interface, addresses in net_interfaces.items():
                interfaces_info[interface] = []
                for addr in addresses:
                    if addr.family.name == 'AF_INET':
                        interfaces_info[interface].append({
                            'address': addr.address,
                            'netmask': addr.netmask
                        })

            # Estadísticas de red
            net_stats = psutil.net_io_counters()
            stats = {
                'bytes_sent': net_stats.bytes_sent,
                'bytes_recv': net_stats.bytes_recv,
                'packets_sent': net_stats.packets_sent,
                'packets_recv': net_stats.packets_recv
            } if net_stats else {}

            return {
                'interfaces': interfaces_info,
                'stats': stats
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def benchmark_cpu() -> Dict[str, Any]:
        """Realiza benchmark simple de CPU."""
        import time

        # Benchmark de cálculo numérico simple
        start_time = time.time()
        result = sum(i * i for i in range(1000000))
        cpu_time = time.time() - start_time

        return {
            'computation_time': cpu_time,
            'result': result,
            'performance_score': 1000000 / cpu_time  # operaciones por segundo
        }

    @staticmethod
    def get_optimal_config() -> Dict[str, Any]:
        """Genera configuración óptima basada en hardware detectado."""
        cpu_info = HardwareDetector.detect_cpu_info()
        mem_info = HardwareDetector.detect_memory_info()
        gpu_info = HardwareDetector.detect_gpu_info()

        # Configuración óptima
        config = {
            'max_workers': min(cpu_info['count'], max(1, cpu_info['count'] // 2)),
            'memory_limit_gb': min(16.0, mem_info['available_gb'] * 0.7),
            'use_gpu': gpu_info['available'],
            'gpu_memory_limit': 4.0 if gpu_info['available'] else 0.0
        }

        # Ajustes específicos para diferentes tipos de sistema
        if cpu_info['count'] >= 16:  # Sistema grande
            config.update({
                'max_workers': cpu_info['count'] // 2,
                'memory_limit_gb': min(32.0, mem_info['available_gb'] * 0.8)
            })
        elif cpu_info['count'] <= 4:  # Sistema pequeño
            config.update({
                'max_workers': max(1, cpu_info['count'] // 2),
                'memory_limit_gb': min(4.0, mem_info['available_gb'] * 0.6)
            })

        return config

    @classmethod
    def get_system_report(cls) -> Dict[str, Any]:
        """Genera reporte completo del sistema."""
        return {
            'cpu': cls.detect_cpu_info(),
            'memory': cls.detect_memory_info(),
            'gpu': cls.detect_gpu_info(),
            'storage': cls.detect_storage_info(),
            'network': cls.detect_network_info(),
            'optimal_config': cls.get_optimal_config(),
            'timestamp': psutil.time.time()
        }


class EnvironmentConfigurator:
    """Configurador de variables de entorno para optimización."""

    @staticmethod
    def configure_openmp(num_threads: int):
        """Configura variables de entorno para OpenMP."""
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

    @staticmethod
    def configure_memory(memory_gb: float):
        """Configura límites de memoria."""
        memory_mb = int(memory_gb * 1024)
        os.environ['PYSCF_MAX_MEMORY'] = str(memory_mb)

    @staticmethod
    def configure_gpu():
        """Configura entorno para GPU."""
        # Configuraciones específicas para gpu4pyscf
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        os.environ.setdefault('PYSCF_GPU_MEMORY', '4096')  # MB

    @staticmethod
    def apply_hardware_config(config_dict: Dict[str, Any]):
        """Aplica configuración de hardware."""
        # Configurar paralelismo
        if 'max_workers' in config_dict:
            EnvironmentConfigurator.configure_openmp(config_dict['max_workers'])

        # Configurar memoria
        if 'memory_limit_gb' in config_dict:
            EnvironmentConfigurator.configure_memory(config_dict['memory_limit_gb'])

        # Configurar GPU
        if config_dict.get('use_gpu', False):
            EnvironmentConfigurator.configure_gpu()

    @staticmethod
    def get_current_config() -> Dict[str, str]:
        """Obtiene configuración actual de entorno."""
        env_vars = [
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'PYSCF_MAX_MEMORY', 'CUDA_VISIBLE_DEVICES', 'PYSCF_GPU_MEMORY'
        ]

        return {var: os.environ.get(var, 'NOT_SET') for var in env_vars}