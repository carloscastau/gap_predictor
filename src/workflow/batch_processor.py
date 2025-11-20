# src/workflow/batch_processor.py
"""Procesador por lotes para ejecución eficiente de múltiples materiales."""

import asyncio
import time
from typing import Dict, Any, List, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
import json

from ..utils.logging import StructuredLogger


logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Modos de procesamiento por lotes."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"


@dataclass
class BatchTask:
    """Tarea individual en un lote."""
    item_id: str
    item_data: Any
    priority: int = 0
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    
    @property
    def duration(self) -> float:
        """Duración de la tarea."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
    
    @property
    def is_completed(self) -> bool:
        """Si la tarea está completada."""
        return self.status in ["completed", "failed", "cancelled"]


@dataclass
class BatchProgress:
    """Progreso de procesamiento por lotes."""
    total_items: int
    completed_items: int
    failed_items: int
    running_items: int
    pending_items: int
    start_time: float
    estimated_remaining_time: Optional[float] = None
    current_throughput: float = 0.0  # items/second
    
    @property
    def progress_percentage(self) -> float:
        """Porcentaje de progreso."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito."""
        if self.completed_items == 0:
            return 0.0
        return (self.completed_items - self.failed_items) / self.completed_items * 100
    
    @property
    def elapsed_time(self) -> float:
        """Tiempo transcurrido."""
        return time.perf_counter() - self.start_time


class BatchProcessor:
    """Procesador inteligente por lotes para múltiples materiales."""
    
    def __init__(self, 
                 max_concurrent: int = 4,
                 processing_mode: ProcessingMode = ProcessingMode.PARALLEL_THREADS,
                 enable_progress_tracking: bool = True,
                 retry_failed_tasks: bool = True,
                 max_retries: int = 3):
        """
        Inicializa el procesador por lotes.
        
        Args:
            max_concurrent: Máximo número de tareas concurrentes
            processing_mode: Modo de procesamiento
            enable_progress_tracking: Habilitar tracking de progreso
            retry_failed_tasks: Reintentar tareas fallidas
            max_retries: Máximo número de reintentos
        """
        self.max_concurrent = max_concurrent
        self.processing_mode = processing_mode
        self.enable_progress_tracking = enable_progress_tracking
        self.retry_failed_tasks = retry_failed_tasks
        self.max_retries = max_retries
        
        self.logger = StructuredLogger("BatchProcessor")
        
        # Estado interno
        self.tasks: Dict[str, BatchTask] = {}
        self.progress: Optional[BatchProgress] = None
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.is_running = False
        self.is_cancelled = False
        
        # Configurar executor
        self._setup_executor()
        
        self.logger.info(f"BatchProcessor inicializado: {max_concurrent} workers, {processing_mode.value}")
    
    def _setup_executor(self):
        """Configura el executor según el modo de procesamiento."""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        if self.processing_mode == ProcessingMode.PARALLEL_THREADS:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        elif self.processing_mode == ProcessingMode.PARALLEL_PROCESSES:
            self.executor = ProcessPoolExecutor(max_workers=self.max_concurrent)
        else:  # SEQUENTIAL
            self.executor = None
    
    async def process_batch(self,
                          items: List[Any],
                          process_func: Callable[[Any], Any],
                          item_id_func: Optional[Callable[[Any], str]] = None,
                          progress_callback: Optional[Callable[[int, int], None]] = None,
                          priority_func: Optional[Callable[[Any], int]] = None) -> List[Any]:
        """
        Procesa un lote de elementos.
        
        Args:
            items: Lista de elementos a procesar
            process_func: Función para procesar cada elemento
            item_id_func: Función para obtener ID de elemento (usa str() si None)
            progress_callback: Callback de progreso (completed, total)
            priority_func: Función para obtener prioridad (usa 0 si None)
            
        Returns:
            Lista de resultados en el mismo orden que los items
        """
        if not items:
            return []
        
        # Configurar IDs y prioridades
        task_items = []
        for i, item in enumerate(items):
            item_id = item_id_func(item) if item_id_func else f"item_{i}"
            priority = priority_func(item) if priority_func else 0
            
            task = BatchTask(
                item_id=item_id,
                item_data=item,
                priority=priority
            )
            task_items.append(task)
            self.tasks[item_id] = task
        
        # Ordenar por prioridad
        task_items.sort(key=lambda t: t.priority, reverse=True)
        
        # Inicializar progreso
        if self.enable_progress_tracking:
            self.progress = BatchProgress(
                total_items=len(task_items),
                completed_items=0,
                failed_items=0,
                running_items=0,
                pending_items=len(task_items),
                start_time=time.perf_counter()
            )
        
        self.logger.info(f"Iniciando procesamiento de {len(task_items)} elementos")
        
        try:
            # Ejecutar según el modo
            if self.processing_mode == ProcessingMode.SEQUENTIAL:
                results = await self._process_sequential(task_items, process_func, progress_callback)
            else:
                results = await self._process_parallel(task_items, process_func, progress_callback)
            
            # Log final
            if self.progress:
                self.logger.info(f"Procesamiento completado: {self.progress.completed_items} exitosos, "
                               f"{self.progress.failed_items} fallidos")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento por lotes: {e}")
            raise
        finally:
            self._cleanup()
    
    async def _process_sequential(self, 
                                tasks: List[BatchTask], 
                                process_func: Callable,
                                progress_callback: Optional[Callable]) -> List[Any]:
        """Procesa tareas secuencialmente."""
        results = []
        
        for i, task in enumerate(tasks):
            if self.is_cancelled:
                break
            
            try:
                task.status = "running"
                task.start_time = time.perf_counter()
                
                result = await self._execute_single_task(task, process_func)
                task.result = result
                task.status = "completed"
                results.append(result)
                
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                results.append(None)
                self.logger.error(f"Error procesando {task.item_id}: {e}")
            
            finally:
                task.end_time = time.perf_counter()
                
                # Actualizar progreso
                if self.enable_progress_tracking:
                    self._update_progress(task)
                
                # Callback de progreso
                if progress_callback:
                    progress_callback(self.progress.completed_items + self.progress.failed_items, 
                                    self.progress.total_items)
        
        return results
    
    async def _process_parallel(self, 
                              tasks: List[BatchTask], 
                              process_func: Callable,
                              progress_callback: Optional[Callable]) -> List[Any]:
        """Procesa tareas en paralelo usando executor."""
        if not self.executor:
            raise RuntimeError("Executor no configurado para procesamiento paralelo")
        
        self.is_running = True
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(task: BatchTask) -> Any:
            async with semaphore:
                if self.is_cancelled:
                    raise asyncio.CancelledError()
                
                task.status = "running"
                task.start_time = time.perf_counter()
                
                try:
                    result = await self._execute_single_task(task, process_func)
                    task.result = result
                    task.status = "completed"
                    return result
                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                    raise
                finally:
                    task.end_time = time.perf_counter()
                    
                    # Actualizar progreso
                    if self.enable_progress_tracking:
                        self._update_progress(task)
                    
                    # Callback de progreso
                    if progress_callback and self.progress:
                        progress_callback(self.progress.completed_items + self.progress.failed_items, 
                                        self.progress.total_items)
        
        # Crear y ejecutar tareas asíncronas
        async_tasks = [execute_with_semaphore(task) for task in tasks]
        
        # Ejecutar con control de cancelación
        try:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Procesar excepciones
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error en tarea {tasks[i].item_id}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.CancelledError:
            self.logger.warning("Procesamiento cancelado por el usuario")
            raise
        except Exception as e:
            self.logger.error(f"Error en procesamiento paralelo: {e}")
            raise
    
    async def _execute_single_task(self, task: BatchTask, process_func: Callable) -> Any:
        """Ejecuta una tarea individual."""
        # Ejecutar función síncrona en thread pool si es necesario
        if asyncio.iscoroutinefunction(process_func):
            return await process_func(task.item_data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, process_func, task.item_data)
    
    def _update_progress(self, completed_task: BatchTask):
        """Actualiza el progreso global."""
        if not self.progress:
            return
        
        # Actualizar contadores
        if completed_task.status == "completed":
            self.progress.completed_items += 1
        elif completed_task.status == "failed":
            self.progress.failed_items += 1
        
        self.progress.pending_items = max(0, self.progress.total_items - 
                                        self.progress.completed_items - self.progress.failed_items)
        
        # Actualizar throughput
        elapsed = self.progress.elapsed_time
        if elapsed > 0:
            completed = self.progress.completed_items + self.progress.failed_items
            self.progress.current_throughput = completed / elapsed
        
        # Estimar tiempo restante
        if self.progress.current_throughput > 0 and self.progress.pending_items > 0:
            self.progress.estimated_remaining_time = self.progress.pending_items / self.progress.current_throughput
    
    def get_progress_status(self) -> Optional[Dict[str, Any]]:
        """Obtiene estado actual del progreso."""
        if not self.enable_progress_tracking or not self.progress:
            return None
        
        return {
            'total_items': self.progress.total_items,
            'completed_items': self.progress.completed_items,
            'failed_items': self.progress.failed_items,
            'running_items': self.progress.running_items,
            'pending_items': self.progress.pending_items,
            'progress_percentage': self.progress.progress_percentage,
            'success_rate': self.progress.success_rate,
            'elapsed_time': self.progress.elapsed_time,
            'current_throughput': self.progress.current_throughput,
            'estimated_remaining_time': self.progress.estimated_remaining_time,
            'estimated_completion': time.strftime('%H:%M:%S', time.localtime(
                time.time() + self.progress.estimated_remaining_time
            )) if self.progress.estimated_remaining_time else None
        }
    
    def stop_processing(self):
        """Detiene el procesamiento en curso."""
        self.is_cancelled = True
        self.logger.info("Procesamiento detenido por solicitud del usuario")
    
    def _cleanup(self):
        """Limpia recursos."""
        self.is_running = False
        self.is_cancelled = False
        
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
    
    def get_task_status(self, item_id: str) -> Optional[BatchTask]:
        """Obtiene estado de una tarea específica."""
        return self.tasks.get(item_id)
    
    def get_failed_tasks(self) -> List[BatchTask]:
        """Obtiene lista de tareas fallidas."""
        return [task for task in self.tasks.values() if task.status == "failed"]
    
    def retry_failed_tasks(self, process_func: Callable) -> List[Any]:
        """Reintenta tareas fallidas."""
        if not self.retry_failed_tasks:
            return []
        
        failed_tasks = self.get_failed_tasks()
        if not failed_tasks:
            return []
        
        self.logger.info(f"Reintentando {len(failed_tasks)} tareas fallidas")
        
        # Filtrar tareas que no excedan el máximo de reintentos
        tasks_to_retry = [task for task in failed_tasks if task.retry_count < self.max_retries]
        
        if not tasks_to_retry:
            self.logger.warning("No hay tareas para reintentar")
            return []
        
        # Ejecutar reintentos (síncrono para simplificar)
        retry_results = []
        for task in tasks_to_retry:
            try:
                task.retry_count += 1
                task.status = "pending"
                task.error = None
                
                # Ejecutar en executor
                if asyncio.iscoroutinefunction(process_func):
                    result = asyncio.run(process_func(task.item_data))
                else:
                    loop = asyncio.get_event_loop()
                    result = loop.run_in_executor(self.executor, process_func, task.item_data)
                
                task.result = result
                task.status = "completed"
                retry_results.append(result)
                
                self.logger.info(f"Tarea {task.item_id} reintentada exitosamente")
                
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                retry_results.append(None)
                self.logger.error(f"Reintento fallido para {task.item_id}: {e}")
        
        return retry_results
    
    def save_progress_report(self, filepath: Path):
        """Guarda reporte de progreso a archivo."""
        if not self.enable_progress_tracking:
            return
        
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'progress': self.get_progress_status(),
            'tasks': {
                task_id: {
                    'status': task.status,
                    'duration': task.duration,
                    'retry_count': task.retry_count,
                    'error': task.error
                }
                for task_id, task in self.tasks.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Reporte de progreso guardado en {filepath}")
    
    def __del__(self):
        """Destructor."""
        self._cleanup()


# Funciones de conveniencia para casos específicos

async def process_materials_parallel(materials: List[str], 
                                   process_func: Callable[[str], Any],
                                   max_workers: int = 4,
                                   progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
    """
    Procesa una lista de materiales en paralelo.
    
    Args:
        materials: Lista de materiales
        process_func: Función para procesar cada material
        max_workers: Número máximo de workers
        progress_callback: Callback de progreso
        
    Returns:
        Lista de resultados
    """
    processor = BatchProcessor(
        max_concurrent=max_workers,
        processing_mode=ProcessingMode.PARALLEL_THREADS,
        enable_progress_tracking=True
    )
    
    return await processor.process_batch(
        items=materials,
        process_func=process_func,
        item_id_func=lambda x: x,
        progress_callback=progress_callback
    )


async def process_materials_sequential(materials: List[str], 
                                     process_func: Callable[[str], Any],
                                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
    """
    Procesa una lista de materiales secuencialmente.
    
    Args:
        materials: Lista de materiales
        process_func: Función para procesar cada material
        progress_callback: Callback de progreso
        
    Returns:
        Lista de resultados
    """
    processor = BatchProcessor(
        processing_mode=ProcessingMode.SEQUENTIAL,
        enable_progress_tracking=True
    )
    
    return await processor.process_batch(
        items=materials,
        process_func=process_func,
        item_id_func=lambda x: x,
        progress_callback=progress_callback
    )


class MaterialBatchProcessor(BatchProcessor):
    """BatchProcessor especializado para materiales semiconductores."""
    
    def __init__(self, 
                 max_concurrent: int = 4,
                 enable_memory_management: bool = True,
                 memory_limit_gb: float = 8.0):
        """
        Inicializa procesador especializado para materiales.
        
        Args:
            max_concurrent: Máximo número de materiales concurrentes
            enable_memory_management: Habilitar gestión de memoria
            memory_limit_gb: Límite de memoria en GB
        """
        super().__init__(
            max_concurrent=max_concurrent,
            processing_mode=ProcessingMode.PARALLEL_THREADS,
            enable_progress_tracking=True
        )
        
        self.enable_memory_management = enable_memory_management
        self.memory_limit_mb = memory_limit_gb * 1024
        
        if enable_memory_management:
            self.logger.info(f"Gestión de memoria habilitada: {memory_limit_gb}GB límite")
    
    async def execute_material_calculation(self, formula: str, calculation_func: Callable) -> Dict[str, Any]:
        """
        Ejecuta cálculo para un material con gestión de memoria.
        
        Args:
            formula: Fórmula del material
            calculation_func: Función de cálculo
            
        Returns:
            Resultado del cálculo
        """
        if self.enable_memory_management:
            # Verificar memoria disponible antes de ejecutar
            available_memory = await self._check_available_memory()
            if available_memory < 1000:  # Menos de 1GB disponible
                self.logger.warning(f"Memoria baja disponible ({available_memory:.0f}MB), "
                                  f"reduc concurrency temporalmente")
                self.max_concurrent = max(1, self.max_concurrent // 2)
        
        try:
            result = await calculation_func(formula)
            return {
                'formula': formula,
                'success': True,
                'result': result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            return {
                'formula': formula,
                'success': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    async def _check_available_memory(self) -> float:
        """Verifica memoria disponible en MB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return 4000  # Valor por defecto si psutil no está disponible