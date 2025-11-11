# src/workflow/checkpoint.py
"""Sistema de checkpoints para recuperación de fallos."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import gzip

from ...utils.logging import StructuredLogger


@dataclass
class CheckpointData:
    """Datos de un checkpoint."""
    stage: str
    timestamp: str
    data: Dict[str, Any]
    status: str = "in_progress"
    version: str = "2.0"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        result = asdict(self)
        if self.metadata is None:
            result['metadata'] = {}
        return result


class CheckpointManager:
    """Gestor de checkpoints para recuperación de fallos."""

    def __init__(self, checkpoint_dir: Path, compress: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.logger = StructuredLogger("CheckpointManager")

    def save_checkpoint(self, stage: str, data: Dict[str, Any],
                       status: str = "in_progress") -> Path:
        """Guarda un checkpoint."""
        timestamp = time.strftime('%Y%m%d_%H%M%S_%f')

        checkpoint_data = CheckpointData(
            stage=stage,
            timestamp=timestamp,
            data=data,
            status=status,
            metadata={
                'hostname': self._get_hostname(),
                'pid': self._get_pid()
            }
        )

        filename = f"checkpoint_{stage}_{timestamp}.json"
        if self.compress:
            filename += ".gz"

        filepath = self.checkpoint_dir / filename

        try:
            checkpoint_dict = checkpoint_data.to_dict()

            if self.compress:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(checkpoint_dict, f, indent=2, default=str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_dict, f, indent=2, default=str)

            self.logger.info(f"Saved checkpoint: {stage} -> {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {stage}: {e}")
            raise

    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Carga el checkpoint más reciente para un stage."""
        checkpoints = self._find_checkpoints(stage)

        if not checkpoints:
            return None

        # Encontrar el más reciente por timestamp
        latest_checkpoint = max(checkpoints, key=lambda x: x['timestamp'])

        try:
            filepath = latest_checkpoint['filepath']

            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            self.logger.info(f"Loaded checkpoint: {stage} <- {filepath}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {stage}: {e}")
            return None

    def save_stage_result(self, stage_name: str, stage_result) -> None:
        """Guarda resultado de un stage."""
        data = {
            'stage_name': stage_result.stage_name,
            'success': stage_result.success,
            'data': stage_result.data,
            'metrics': stage_result.metrics,
            'duration': stage_result.duration,
            'timestamp': stage_result.timestamp
        }

        self.save_checkpoint(f"stage_{stage_name}", data, "completed")

    def load_stage_result(self, stage_name: str):
        """Carga resultado de un stage."""
        checkpoint = self.load_checkpoint(f"stage_{stage_name}")
        if checkpoint:
            from .pipeline import StageResult
            data = checkpoint['data']
            return StageResult(
                success=data['success'],
                data=data['data'],
                metrics=data['metrics'],
                duration=data['duration'],
                timestamp=data['timestamp'],
                stage_name=data['stage_name']
            )
        return None

    def save_progress(self, completed_stages: list) -> None:
        """Guarda progreso general del pipeline."""
        data = {
            'completed_stages': completed_stages,
            'total_stages': len(completed_stages),  # Se actualizará al cargar
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'progress_percentage': 0  # Se calculará al cargar
        }

        self.save_checkpoint("pipeline_progress", data, "progress")

    def load_progress(self) -> Dict[str, Any]:
        """Carga progreso del pipeline."""
        checkpoint = self.load_checkpoint("pipeline_progress")
        if checkpoint:
            data = checkpoint['data']
            completed = data.get('completed_stages', [])
            # Asumir 3 stages por defecto (cutoff, kmesh, lattice)
            total = 3
            return {
                'completed_stages': completed,
                'total_stages': total,
                'progress_percentage': len(completed) / total * 100,
                'last_updated': data.get('last_updated', 'unknown')
            }

        return {
            'completed_stages': [],
            'total_stages': 3,
            'progress_percentage': 0,
            'last_updated': 'never'
        }

    def list_checkpoints(self, stage: Optional[str] = None) -> list:
        """Lista todos los checkpoints disponibles."""
        return self._find_checkpoints(stage)

    def cleanup_old_checkpoints(self, keep_recent: int = 5) -> int:
        """Limpia checkpoints antiguos, manteniendo los más recientes."""
        all_checkpoints = []

        for pattern in ["*.json", "*.json.gz"]:
            all_checkpoints.extend(self.checkpoint_dir.glob(pattern))

        if len(all_checkpoints) <= keep_recent:
            return 0

        # Ordenar por tiempo de modificación (más reciente primero)
        all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Eliminar los antiguos
        to_delete = all_checkpoints[keep_recent:]
        deleted_count = 0

        for checkpoint in to_delete:
            try:
                checkpoint.unlink()
                deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete checkpoint {checkpoint}: {e}")

        self.logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count

    def _find_checkpoints(self, stage: Optional[str] = None) -> list:
        """Encuentra checkpoints para un stage específico."""
        checkpoints = []

        for pattern in ["*.json", "*.json.gz"]:
            for filepath in self.checkpoint_dir.glob(pattern):
                try:
                    # Leer metadata básica sin cargar todo el archivo
                    if filepath.suffix == '.gz':
                        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                            first_line = f.readline()
                            if '"stage":' in first_line:
                                # Extraer stage del JSON
                                data = json.loads(f.read() + '}')
                                cp_stage = data.get('stage')
                            else:
                                continue
                    else:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            cp_stage = data.get('stage')

                    if stage is None or cp_stage == stage:
                        checkpoints.append({
                            'filepath': filepath,
                            'stage': cp_stage,
                            'timestamp': data.get('timestamp', ''),
                            'status': data.get('status', 'unknown'),
                            'size': filepath.stat().st_size
                        })

                except Exception:
                    continue

        return checkpoints

    def _get_hostname(self) -> str:
        """Obtiene nombre del host."""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"

    def _get_pid(self) -> int:
        """Obtiene ID del proceso."""
        import os
        return os.getpid()

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de checkpoints."""
        all_checkpoints = self.list_checkpoints()

        stats = {
            'total_checkpoints': len(all_checkpoints),
            'stages': {},
            'total_size_mb': 0,
            'oldest_checkpoint': None,
            'newest_checkpoint': None
        }

        timestamps = []

        for cp in all_checkpoints:
            stage = cp['stage']
            if stage not in stats['stages']:
                stats['stages'][stage] = 0
            stats['stages'][stage] += 1

            stats['total_size_mb'] += cp['size'] / (1024 * 1024)

            if cp['timestamp']:
                timestamps.append(cp['timestamp'])

        if timestamps:
            stats['oldest_checkpoint'] = min(timestamps)
            stats['newest_checkpoint'] = max(timestamps)

        return stats


# Funciones de utilidad para compatibilidad con código existente
def save_checkpoint(out_root: Path, stage: str, data: Dict[str, Any], log=None):
    """Función de compatibilidad con código existente."""
    manager = CheckpointManager(out_root / "checkpoints")
    filepath = manager.save_checkpoint(stage, data)

    if log:
        log.info(f"[CHECKPOINT] Saved: {stage} -> {filepath}")

    return filepath


def load_latest_checkpoint(out_root: Path, stage: str = None) -> Dict[str, Any]:
    """Función de compatibilidad para cargar checkpoint."""
    manager = CheckpointManager(out_root / "checkpoints")

    if stage:
        checkpoint = manager.load_checkpoint(stage)
        return checkpoint if checkpoint else {}
    else:
        # Encontrar el más reciente de cualquier stage
        all_checkpoints = manager.list_checkpoints()
        if all_checkpoints:
            # Ordenar por timestamp
            latest = max(all_checkpoints, key=lambda x: x['timestamp'])
            return manager.load_checkpoint(latest['stage']) or {}
        return {}