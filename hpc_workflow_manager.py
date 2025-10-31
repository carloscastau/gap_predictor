#!/usr/bin/env python3
"""
Gestor de flujos de trabajo HPC para cálculos DFT distribuidos.
Coordina múltiples trabajos SLURM y consolida resultados.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class HPCWorkflow:
    """Configuración de flujo de trabajo HPC."""
    name: str
    description: str
    job_scripts: List[str]
    dependencies: List[str] = None
    max_runtime_hours: int = 24
    priority: str = "normal"
    email_notifications: bool = True

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class HPCWorkflowManager:
    """Gestor de flujos de trabajo HPC."""

    def __init__(self, workspace_dir: Path = Path(".")):
        self.workspace_dir = workspace_dir
        self.workflows: Dict[str, HPCWorkflow] = {}
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_workflows: List[str] = []

    def register_workflow(self, workflow: HPCWorkflow):
        """Registra un nuevo flujo de trabajo."""
        self.workflows[workflow.name] = workflow
        print(f"✅ Workflow registrado: {workflow.name}")

    def submit_workflow(self, workflow_name: str) -> Optional[str]:
        """Envía un flujo de trabajo completo."""
        if workflow_name not in self.workflows:
            print(f"❌ Workflow no encontrado: {workflow_name}")
            return None

        workflow = self.workflows[workflow_name]
        print(f"🚀 Enviando workflow: {workflow_name}")
        print(f"📝 Descripción: {workflow.description}")

        submitted_jobs = []
        previous_job_id = None

        for script_path in workflow.job_scripts:
            script_full_path = self.workspace_dir / script_path

            if not script_full_path.exists():
                print(f"❌ Script no encontrado: {script_path}")
                return None

            # Construir comando sbatch
            cmd = ["sbatch"]

            # Agregar dependencias
            if previous_job_id and workflow.dependencies:
                cmd.extend(["--dependency", f"afterok:{previous_job_id}"])

            # Configuración adicional
            if workflow.max_runtime_hours:
                cmd.extend(["--time", f"{workflow.max_runtime_hours}:00:00"])

            if workflow.priority != "normal":
                cmd.extend(["--priority", workflow.priority])

            cmd.append(str(script_full_path))

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]

                submitted_jobs.append({
                    "script": script_path,
                    "job_id": job_id,
                    "submitted_at": datetime.now().isoformat()
                })

                previous_job_id = job_id
                print(f"✅ Job enviado: {script_path} → {job_id}")

            except subprocess.CalledProcessError as e:
                print(f"❌ Error enviando {script_path}: {e}")
                return None

        # Registrar workflow activo
        self.active_jobs[workflow_name] = {
            "workflow": asdict(workflow),
            "jobs": submitted_jobs,
            "started_at": datetime.now().isoformat(),
            "status": "running"
        }

        self._save_state()
        return workflow_name

    def monitor_workflows(self):
        """Monitorea el estado de los workflows activos."""
        for workflow_name, workflow_data in self.active_jobs.items():
            if workflow_data["status"] == "running":
                self._check_workflow_status(workflow_name)

    def _check_workflow_status(self, workflow_name: str):
        """Verifica el estado de un workflow."""
        workflow_data = self.active_jobs[workflow_name]
        all_completed = True

        for job_info in workflow_data["jobs"]:
            job_id = job_info["job_id"]

            try:
                # Consultar estado del job
                result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"],
                                      capture_output=True, text=True)

                if result.returncode == 0 and result.stdout.strip():
                    status = result.stdout.strip()
                    job_info["current_status"] = status

                    if status not in ["PENDING", "RUNNING", "COMPLETING"]:
                        job_info["completed_at"] = datetime.now().isoformat()
                    else:
                        all_completed = False
                else:
                    # Job no encontrado, asumir completado
                    job_info["current_status"] = "COMPLETED"
                    job_info["completed_at"] = datetime.now().isoformat()

            except subprocess.CalledProcessError:
                job_info["current_status"] = "UNKNOWN"
                all_completed = False

        if all_completed:
            workflow_data["status"] = "completed"
            workflow_data["completed_at"] = datetime.now().isoformat()
            self.completed_workflows.append(workflow_name)
            print(f"✅ Workflow completado: {workflow_name}")

        self._save_state()

    def get_workflow_status(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado detallado de un workflow."""
        if workflow_name in self.active_jobs:
            return self.active_jobs[workflow_name]
        return None

    def list_workflows(self) -> Dict[str, List[str]]:
        """Lista todos los workflows disponibles y activos."""
        return {
            "available": list(self.workflows.keys()),
            "active": list(self.active_jobs.keys()),
            "completed": self.completed_workflows
        }

    def cancel_workflow(self, workflow_name: str) -> bool:
        """Cancela un workflow activo."""
        if workflow_name not in self.active_jobs:
            print(f"❌ Workflow no activo: {workflow_name}")
            return False

        workflow_data = self.active_jobs[workflow_name]

        for job_info in workflow_data["jobs"]:
            job_id = job_info["job_id"]
            try:
                subprocess.run(["scancel", job_id], check=True)
                print(f"🛑 Job cancelado: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error cancelando {job_id}: {e}")

        workflow_data["status"] = "cancelled"
        workflow_data["cancelled_at"] = datetime.now().isoformat()

        self._save_state()
        return True

    def _save_state(self):
        """Guarda el estado actual del gestor."""
        state_file = self.workspace_dir / "hpc_workflow_state.json"
        state = {
            "active_jobs": self.active_jobs,
            "completed_workflows": self.completed_workflows,
            "last_updated": datetime.now().isoformat()
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Carga el estado guardado."""
        state_file = self.workspace_dir / "hpc_workflow_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.active_jobs = state.get("active_jobs", {})
                self.completed_workflows = state.get("completed_workflows", [])
            except Exception as e:
                print(f"⚠️ Error cargando estado: {e}")


def create_predefined_workflows() -> List[HPCWorkflow]:
    """Crea workflows predefinidos para cálculos DFT."""

    workflows = []

    # Workflow básico de preconvergencia
    basic_workflow = HPCWorkflow(
        name="gaas_preconvergence_basic",
        description="Preconvergencia básica de GaAs en un solo nodo",
        job_scripts=["slurm_job.sh"],
        max_runtime_hours=12,
        priority="normal"
    )
    workflows.append(basic_workflow)

    # Workflow de barrido paramétrico
    parametric_workflow = HPCWorkflow(
        name="gaas_parametric_sweep",
        description="Barrido paramétrico con array jobs",
        job_scripts=["slurm_array_job.sh"],
        max_runtime_hours=24,
        priority="normal"
    )
    workflows.append(parametric_workflow)

    # Workflow multi-nodo avanzado
    advanced_workflow = HPCWorkflow(
        name="gaas_multi_node_advanced",
        description="Cálculo avanzado multi-nodo con optimización",
        job_scripts=["slurm_multi_node.sh"],
        dependencies=["afterok"],
        max_runtime_hours=48,
        priority="high"
    )
    workflows.append(advanced_workflow)

    # Workflow completo: preconvergencia → optimización → análisis
    full_workflow = HPCWorkflow(
        name="gaas_full_pipeline",
        description="Pipeline completo: preconvergencia, optimización y análisis",
        job_scripts=[
            "slurm_job.sh",
            "slurm_array_job.sh",
            "slurm_multi_node.sh"
        ],
        dependencies=["afterok", "afterok"],
        max_runtime_hours=72,
        priority="high"
    )
    workflows.append(full_workflow)

    return workflows


def main():
    """Interfaz de línea de comandos para el gestor de workflows."""
    import argparse

    parser = argparse.ArgumentParser(description="Gestor de flujos de trabajo HPC para DFT")
    parser.add_argument("action", choices=["submit", "monitor", "status", "list", "cancel"],
                       help="Acción a realizar")
    parser.add_argument("--workflow", help="Nombre del workflow")
    parser.add_argument("--workspace", type=Path, default=Path("."),
                       help="Directorio de trabajo")

    args = parser.parse_args()

    manager = HPCWorkflowManager(args.workspace)
    manager._load_state()

    # Registrar workflows predefinidos
    for workflow in create_predefined_workflows():
        manager.register_workflow(workflow)

    if args.action == "submit":
        if not args.workflow:
            print("❌ Especificar --workflow para submit")
            sys.exit(1)

        result = manager.submit_workflow(args.workflow)
        if result:
            print(f"🚀 Workflow enviado exitosamente: {result}")

    elif args.action == "monitor":
        manager.monitor_workflows()
        print("📊 Monitoreo completado")

    elif args.action == "status":
        if not args.workflow:
            print("❌ Especificar --workflow para status")
            sys.exit(1)

        status = manager.get_workflow_status(args.workflow)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"❌ Workflow no encontrado: {args.workflow}")

    elif args.action == "list":
        workflows = manager.list_workflows()
        print("📋 Workflows disponibles:")
        for category, items in workflows.items():
            print(f"  {category.title()}: {', '.join(items) if items else 'Ninguno'}")

    elif args.action == "cancel":
        if not args.workflow:
            print("❌ Especificar --workflow para cancel")
            sys.exit(1)

        if manager.cancel_workflow(args.workflow):
            print(f"🛑 Workflow cancelado: {args.workflow}")


if __name__ == "__main__":
    main()