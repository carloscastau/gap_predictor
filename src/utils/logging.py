# src/utils/logging.py
"""Logger estructurado con métricas de rendimiento."""

import logging
import time
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class StructuredLogger:
    """Logger estructurado con métricas de rendimiento."""

    def __init__(self, name: str, level: int = logging.INFO, log_dir: Path = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Crear directorio de logs si no existe
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Handler para archivo con rotación
        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(self._get_detailed_formatter())

        # Handler para consola con colores
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_colored_formatter())

        # Evitar duplicados
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _get_detailed_formatter(self) -> logging.Formatter:
        """Formateador detallado para archivos."""
        return logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_colored_formatter(self) -> logging.Formatter:
        """Formateador coloreado para consola."""
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',     # Cyan
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'CRITICAL': '\033[35m',  # Magenta
                'RESET': '\033[0m'       # Reset
            }

            def format(self, record):
                color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                reset = self.COLORS['RESET']

                # Formato simplificado para consola
                message = f"{color}{record.levelname}{reset} | {record.getMessage()}"

                # Agregar información de función para errores
                if record.levelno >= logging.ERROR:
                    message += f" ({record.funcName}:{record.lineno})"

                return message

        return ColoredFormatter()

    @contextmanager
    def operation_timer(self, operation_name: str, log_start: bool = True):
        """Context manager para timing de operaciones."""
        if log_start:
            self.logger.info(f"Starting operation: {operation_name}")

        start_time = time.perf_counter()
        start_wall_time = time.time()

        try:
            yield
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s")

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.logger.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
            raise

    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Loggea métricas de rendimiento."""
        metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self.logger.info(f"PERFORMANCE | {operation} | {metrics_str}")

    def log_calculation_result(self, calc_type: str, parameters: Dict[str, Any],
                              result: Any, duration: float):
        """Loggea resultado de cálculo."""
        params_str = " | ".join(f"{k}={v}" for k, v in parameters.items())
        self.logger.info(f"CALCULATION | {calc_type} | {params_str} | result={result} | time={duration:.3f}s")

    def log_convergence_info(self, parameter: str, values: list, energies: list,
                           converged: bool, optimal_value: Optional[float] = None):
        """Loggea información de convergencia."""
        n_points = len(values)
        energy_range = max(energies) - min(energies) if energies else 0

        status = "CONVERGED" if converged else "NOT_CONVERGED"
        optimal_str = f" | optimal={optimal_value}" if optimal_value else ""

        self.logger.info(f"CONVERGENCE | {parameter} | {status} | points={n_points} | "
                        f"range={energy_range:.2e}{optimal_str}")

    def log_stage_progress(self, stage_name: str, progress: float, message: str = ""):
        """Loggea progreso de un stage."""
        progress_str = f"{progress:.1f}%" if isinstance(progress, (int, float)) else str(progress)
        msg = f" [{message}]" if message else ""
        self.logger.info(f"STAGE | {stage_name} | {progress_str}{msg}")

    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Loggea error con contexto adicional."""
        context_str = ""
        if context:
            context_items = []
            for k, v in context.items():
                if isinstance(v, (int, float)):
                    context_items.append(f"{k}={v}")
                else:
                    context_items.append(f"{k}={str(v)[:50]}...")
            context_str = f" | context: {' | '.join(context_items)}"

        self.logger.error(f"ERROR | {type(error).__name__}: {error}{context_str}")

    def log_system_info(self, info: Dict[str, Any]):
        """Loggea información del sistema."""
        for key, value in info.items():
            if isinstance(value, dict):
                self.logger.info(f"SYSTEM | {key}: {value}")
            else:
                self.logger.info(f"SYSTEM | {key} = {value}")


class PerformanceTracker:
    """Tracker de rendimiento para operaciones."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.operations = {}

    def start_operation(self, name: str) -> str:
        """Inicia tracking de una operación."""
        operation_id = f"{name}_{time.time()}"
        self.operations[operation_id] = {
            'name': name,
            'start_time': time.perf_counter(),
            'start_wall_time': time.time(),
            'status': 'running'
        }
        return operation_id

    def end_operation(self, operation_id: str, result: Any = None, error: Exception = None):
        """Finaliza tracking de una operación."""
        if operation_id not in self.operations:
            self.logger.warning(f"Operation {operation_id} not found in tracker")
            return

        op = self.operations[operation_id]
        end_time = time.perf_counter()
        duration = end_time - op['start_time']

        op.update({
            'end_time': end_time,
            'duration': duration,
            'result': result,
            'error': error,
            'status': 'completed' if error is None else 'failed'
        })

        # Loggear resultado
        if error:
            self.logger.log_performance_metrics(
                op['name'],
                {'duration': duration, 'status': 'failed', 'error': str(error)}
            )
        else:
            self.logger.log_performance_metrics(
                op['name'],
                {'duration': duration, 'status': 'completed', 'result': str(result)[:100]}
            )

    @contextmanager
    def track(self, name: str):
        """Context manager para tracking automático."""
        op_id = self.start_operation(name)
        try:
            yield
            self.end_operation(op_id)
        except Exception as e:
            self.end_operation(op_id, error=e)
            raise


# Configuración global de logging
def setup_global_logging(level: int = logging.INFO, log_dir: Path = None) -> StructuredLogger:
    """Configura logging global para la aplicación."""
    if log_dir is None:
        log_dir = Path("logs")

    # Logger principal
    main_logger = StructuredLogger("preconv", level=level, log_dir=log_dir)

    # Configurar logging de bibliotecas externas
    logging.getLogger('pyscf').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('scipy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    return main_logger


# Funciones de utilidad
def get_logger(name: str) -> StructuredLogger:
    """Obtiene un logger estructurado."""
    return StructuredLogger(name)


def log_system_startup():
    """Loggea información de inicio del sistema."""
    import platform
    import psutil

    logger = get_logger("system")

    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'hostname': platform.node()
    }

    logger.log_system_info(system_info)


def create_operation_logger(operation_name: str) -> StructuredLogger:
    """Crea logger específico para una operación."""
    return StructuredLogger(f"operation.{operation_name}")