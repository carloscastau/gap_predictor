# Guía de Despliegue en Producción - Preconvergencia-GaAs v2.0
## Manual Completo de Implementación y Mantenimiento

**Versión del Documento:** 2.0  
**Fecha:** 2025-11-11  
**Consultor:** Kilo Code - Especialista en Despliegue de Producción

---

## 📋 ÍNDICE

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema de Producción](#arquitectura-del-sistema-de-producción)
3. [Pre-requisitos del Sistema](#pre-requisitos-del-sistema)
4. [Instalación y Configuración](#instalación-y-configuración)
5. [Validación Pre-Despliegue](#validación-pre-despliegue)
6. [Despliegue en Producción](#despliegue-en-producción)
7. [Monitoreo y Alertas](#monitoreo-y-alertas)
8. [Mantenimiento y Operaciones](#mantenimiento-y-operaciones)
9. [Troubleshooting](#troubleshooting)
10. [Procedimientos de Recuperación](#procedimientos-de-recuperación)
11. [Métricas de Éxito](#métricas-de-éxito)
12. [Apéndices](#apéndices)

---

## 1. RESUMEN EJECUTIVO

### 1.1 Objetivo
Este documento proporciona una guía completa para desplegar y operar el sistema Preconvergencia-GaAs v2.0 en un entorno de producción, con énfasis en:

- **Disponibilidad**: 99.5% de uptime
- **Rendimiento**: Optimización de recursos de cálculo
- **Monitoreo**: Supervisión en tiempo real del sistema
- **Recuperación**: Protocolos automáticos de recuperación ante fallos

### 1.2 Componentes Implementados
- ✅ **EnvironmentValidator**: Validación completa del entorno
- ✅ **ProductionMonitor**: Monitoreo en tiempo real
- ✅ **Pipeline Integrado**: Ejecución con monitoreo automático
- ✅ **Sistema de Alertas**: Notificaciones proactivas
- ✅ **Backup y Recovery**: Protección de datos y rollback

### 1.3 Beneficios del Despliegue
- **Detección temprana** de problemas de rendimiento
- **Recuperación automática** de fallos menores
- **Optimización** continua basada en métricas
- **Trazabilidad** completa de ejecuciones
- **Escalabilidad** para cargas de trabajo mayores

---

## 2. ARQUITECTURA DEL SISTEMA DE PRODUCCIÓN

### 2.1 Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCCIÓN STACK                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Validation │  │ Monitoring  │  │   Pipeline  │        │
│  │   System    │  │   System    │  │   System    │        │
│  │             │  │             │  │             │        │
│  │ • Env Check │  │ • Real-time │  │ • DFT Calc  │        │
│  │ • Dep Test  │  │ • Alerts    │  │ • Converg   │        │
│  │ • Benchmk   │  │ • Metrics   │  │ • Optimiz   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Backup    │  │  Recovery   │  │  Logging    │        │
│  │   System    │  │   System    │  │   System    │        │
│  │             │  │             │  │             │        │
│  │ • Auto Snap │  │ • Auto Fix  │  │ • Structured│        │
│  │ • Versioned │  │ • Rollback  │  │ • Levels    │        │
│  │ • Config    │  │ • Restore   │  │ • Rotation  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Flujo de Datos
1. **Validación** → Verificar entorno antes de ejecución
2. **Ejecución** → Pipeline con monitoreo continuo
3. **Monitoreo** → Recolección de métricas en tiempo real
4. **Alertas** → Notificación automática de problemas
5. **Backup** → Protección automática de datos
6. **Recovery** → Recuperación automática de fallos

---

## 3. PRE-REQUISITOS DEL SISTEMA

### 3.1 Requisitos de Hardware

#### Mínimo Recomendado
- **CPU**: 8 cores (x86_64)
- **RAM**: 16 GB
- **Storage**: 100 GB libres (SSD preferible)
- **Network**: Conexión estable para instalación de dependencias

#### Recomendado para Producción
- **CPU**: 16+ cores (x86_64)
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD
- **Network**: Ancho de banda > 10 Mbps

### 3.2 Requisitos de Software

#### Sistema Operativo
- **Ubuntu 20.04+ / Debian 11+**
- **CentOS 8+ / RHEL 8+**
- **SUSE Linux Enterprise 15+**

#### Herramientas de Compilación
```bash
# Ubuntu/Debian
sudo apt-get install -y gfortran gcc cmake build-essential

# CentOS/RHEL  
sudo yum install -y gcc-gfortran gcc cmake make

# Verificar instalación
gfortran --version  # >= 9.0
gcc --version       # >= 9.0
cmake --version     # >= 3.16
```

#### Librerías del Sistema
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    pkg-config

# CentOS/RHEL
sudo yum install -y \
    openblas-devel \
    lapack-devel \
    openmpi-devel
```

### 3.3 Python y Dependencias
- **Python**: 3.9+ (recomendado 3.11)
- **pip**: 21.0+
- **virtualenv**: 20.0+

---

## 4. INSTALACIÓN Y CONFIGURACIÓN

### 4.1 Preparación del Entorno

#### 4.1.1 Crear Estructura de Directorios
```bash
# Crear estructura principal
mkdir -p ~/preconvergencia-gaas
cd ~/preconvergencia-gaas

# Crear subdirectorios
mkdir -p {src,config,scripts,monitoring,logs,backups}
```

#### 4.1.2 Configurar Variables de Entorno
```bash
# Agregar a ~/.bashrc
export PRECONV_HOME=~/preconvergencia-gaas
export PYTHONPATH=$PRECONV_HOME/src:$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Aplicar cambios
source ~/.bashrc
```

### 4.2 Instalación de Dependencias

#### 4.2.1 Crear Entorno Virtual
```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel
```

#### 4.2.2 Instalar Dependencias Científicas
```bash
# Instalar dependencias principales
pip install numpy>=1.24,<2.0
pip install scipy>=1.13,<1.14  
pip install pandas>=1.5,<2.2
pip install matplotlib>=3.7,<3.9
pip install pymatgen>=2024.9.3
pip install spglib>=2.0.2

# Instalar PySCF (puede tardar varios minutos)
pip install pyscf==2.3.0 --no-binary=pyscf
```

#### 4.2.3 Instalar Dependencias de Monitoreo
```bash
# Instalar herramientas de monitoreo
pip install psutil>=5.9.0
pip install schedule>=1.2.0

# Instalar herramientas de desarrollo (opcional)
pip install pytest>=7.0.0
pip install pytest-asyncio>=0.21.0
pip install pytest-cov>=4.0.0
```

### 4.3 Configuración del Sistema

#### 4.3.1 Configuración de Producción
```yaml
# config/production.yaml
lattice_constant: 5.653
x_ga: 0.25
sigma_ha: 0.01

basis_set: "gth-dzvp"
pseudopotential: "gth-pbe"
xc_functional: "PBE"

cutoff_list: [80, 120, 160, 200]
kmesh_list: [(2,2,2), (4,4,4), (6,6,6), (8,8,8)]

max_workers: 8
timeout_seconds: 900
stage_timeout: 1800
memory_limit_gb: 16.0

output_dir: "results"
checkpoint_interval: 30

log_level: "INFO"
log_file: "production.log"

use_gpu: false
gpu_memory_limit: 4.0
```

#### 4.3.2 Configuración de Alertas
```python
# config/alert_config.py
ALERT_CONFIG = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.company.com',
        'smtp_port': 587,
        'from': 'alerts@company.com',
        'to': ['admin@company.com', 'support@company.com'],
        'username': 'alerts@company.com',
        'password': 'secure_password'
    },
    'thresholds': {
        'memory_percent': 85.0,
        'cpu_percent': 90.0,
        'disk_usage_percent': 90.0,
        'convergence_failure_rate': 0.2
    }
}
```

---

## 5. VALIDACIÓN PRE-DESPLIEGUE

### 5.1 Ejecutar Validación Completa

#### 5.1.1 Script de Validación Automática
```bash
# Ejecutar validación completa
./scripts/validate_production_environment.sh

# Opciones avanzadas
./scripts/validate_production_environment.sh --verbose
./scripts/validate_production_environment.sh --quick  # Validación rápida
```

#### 5.1.2 Validación Manual
```bash
# 1. Verificar herramientas de compilación
gfortran --version
gcc --version
cmake --version

# 2. Verificar PySCF
python3 -c "from pyscf.pbc import gto, dft; print('PySCF OK')"

# 3. Ejecutar validador Python
python3 src/utils/environment_validator.py

# 4. Test de configuración
python3 -c "
from config.settings import PreconvergenceConfig
config = PreconvergenceConfig()
print(f'Config loaded: a={config.lattice_constant} Å')
"

# 5. Test del monitor
python3 src/utils/production_monitor.py
```

### 5.2 Checklist de Validación

#### ✅ Entorno Base
- [ ] Python 3.9+ instalado
- [ ] Entorno virtual configurado
- [ ] Dependencias instaladas correctamente
- [ ] PySCF compilado y funcional

#### ✅ Herramientas de Compilación
- [ ] Gfortran 9.0+ disponible
- [ ] GCC 9.0+ disponible  
- [ ] CMake 3.16+ disponible
- [ ] Make disponible

#### ✅ Librerías del Sistema
- [ ] BLAS/LAPACK configuradas
- [ ] OpenMPI configurado
- [ ] Variables de entorno optimizadas

#### ✅ Configuración
- [ ] Archivos de configuración válidos
- [ ] Directorios de salida accesibles
- [ ] Permisos de archivos correctos

#### ✅ Pruebas Funcionales
- [ ] Import de módulos principal
- [ ] Configuración de pipeline
- [ ] Sistema de monitoreo
- [ ] Validador de entorno

---

## 6. DESPLIEGUE EN PRODUCCIÓN

### 6.1 Preparación del Despliegue

#### 6.1.1 Crear Backup Pre-Despliegue
```bash
# Crear backup del sistema actual
tar -czf backup_pre_deployment_$(date +%Y%m%d_%H%M%S).tar.gz \
    src/ config/ scripts/ requirements.txt

# Verificar backup
ls -la backup_pre_deployment_*.tar.gz
```

#### 6.1.2 Configurar Logging
```python
# config/logging_config.py
import logging
from logging.handlers import RotatingFileHandler

# Configurar logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/production.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
```

### 6.2 Iniciar Servicios de Producción

#### 6.2.1 Script de Inicio
```bash
#!/bin/bash
# scripts/start_production.sh

set -e

echo "🚀 Iniciando Preconvergencia-GaAs v2.0 en producción..."

# Activar entorno virtual
source venv/bin/activate

# Validar entorno
if ! python3 src/utils/environment_validator.py; then
    echo "❌ Validación de entorno falló"
    exit 1
fi

# Crear directorios necesarios
mkdir -p {results,monitoring,logs,backups}

# Iniciar pipeline con monitoreo
python3 -c "
import asyncio
from config.settings import PreconvergenceConfig
from workflow.pipeline import PreconvergencePipeline

async def main():
    config = PreconvergenceConfig()
    pipeline = PreconvergencePipeline(config, enable_monitoring=True)
    
    # Verificar requisitos del sistema
    requirements = pipeline.get_system_requirements_check()
    if requirements['overall_status'] != 'ready':
        print('⚠️  Requisitos del sistema:')
        for rec in requirements['recommendations']:
            print(f'  - {rec}')
    
    # Ejecutar pipeline
    result = await pipeline.execute()
    print(f'Resultado: {result.success}')
    if not result.success:
        print(f'Error: {result.error_message}')
    
    # Exportar métricas finales
    pipeline.export_monitoring_data()

asyncio.run(main())
"

echo "✅ Pipeline ejecutado"
```

#### 6.2.2 Configurar como Servicio del Sistema
```ini
# /etc/systemd/system/preconvergencia-gaas.service
[Unit]
Description=Preconvergencia-GaAs v2.0 Production Pipeline
After=network.target
Wants=network.target

[Service]
Type=simple
User=preconv
Group=preconv
WorkingDirectory=/home/preconv/preconvergencia-gaas
Environment=PATH=/home/preconv/preconvergencia-gaAs/venv/bin
ExecStart=/home/preconv/preconvergencia-gaAs/venv/bin/python scripts/start_production.sh
Restart=always
RestartSec=10

# Seguridad
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/home/preconv/preconvergencia-gaAs/results /home/preconv/preconvergencia-gaAs/monitoring

[Install]
WantedBy=multi-user.target
```

### 6.3 Verificar Despliegue

#### 6.3.1 Health Check
```bash
# Verificar estado del servicio
systemctl status preconvergencia-gaas

# Verificar logs
journalctl -u preconvergencia-gaas -f

# Verificar métricas
ls -la monitoring/
tail -f logs/production.log
```

#### 6.3.2 Test de Funcionalidad
```python
# Test rápido de funcionalidad
python3 -c "
import asyncio
from config.settings import get_fast_config
from workflow.pipeline import PreconvergencePipeline

async def quick_test():
    config = get_fast_config()
    pipeline = PreconvergencePipeline(config, enable_monitoring=True)
    
    # Verificar que el monitor funciona
    status = pipeline.get_monitoring_status()
    print(f'Monitor status: {status[\"monitoring_active\"]}')
    
    # Verificar requisitos del sistema
    requirements = pipeline.get_system_requirements_check()
    print(f'System status: {requirements[\"overall_status\"]}')

asyncio.run(quick_test())
"
```

---

## 7. MONITOREO Y ALERTAS

### 7.1 Dashboard de Monitoreo

#### 7.1.1 Acceso al Dashboard
```bash
# Generar reporte HTML
python3 -c "
from config.settings import PreconvergenceConfig
from src.utils.production_monitor import create_production_monitor
import asyncio

async def generate_dashboard():
    config = PreconvergenceConfig()
    monitor = create_production_monitor(config)
    
    # Simular datos de monitoreo
    await asyncio.sleep(2)
    
    # Generar dashboard
    dashboard = monitor.generate_real_time_dashboard(monitor)
    print(f'Dashboard generated: {dashboard}')
    
    # Generar reporte HTML
    html_report = monitor.generate_html_report()
    print(f'HTML report: {html_report}')

asyncio.run(generate_dashboard())
"
```

#### 7.1.2 Métricas Clave
- **CPU Usage**: < 80% promedio
- **Memory Usage**: < 85% de límite configurado
- **Convergence Rate**: > 90%
- **Stage Duration**: Dentro de límites configurados
- **System Health Score**: > 70%

### 7.2 Sistema de Alertas

#### 7.2.1 Configuración de Alertas
```python
# Configurar callbacks de alerta
def email_alert_callback(alert):
    # Enviar email con detalles de la alerta
    print(f"📧 EMAIL ALERT: {alert['message']}")

def webhook_alert_callback(alert):
    # Enviar a webhook de Slack/Teams
    print(f"🔗 WEBHOOK ALERT: {alert['message']}")

# Agregar callbacks al monitor
monitor.add_alert_callback(email_alert_callback)
monitor.add_alert_callback(webhook_alert_callback)
```

#### 7.2.2 Tipos de Alertas

##### Alertas Críticas 🚨
- **System Overload**: CPU + Memory > 80%
- **Stage Timeout**: Stage excede tiempo límite
- **Convergence Failure**: > 20% fallos de convergencia
- **Disk Space**: < 10% espacio libre

##### Alertas de Advertencia ⚠️
- **High CPU**: CPU > 80% por > 5 min
- **High Memory**: Memory > 85%
- **Performance Degradation**: Tiempo de stage > 150% promedio

### 7.3 Reportes Automáticos

#### 7.3.1 Reporte Diario
```bash
# Generar reporte diario
python3 scripts/generate_daily_report.py

# Reporte incluye:
# - Resumen de ejecuciones
# - Métricas de rendimiento
# - Alertas generadas
# - Recomendaciones de optimización
```

#### 7.3.2 Reporte Semanal
```bash
# Análisis semanal
python3 scripts/generate_weekly_analysis.py

# Incluye:
# - Tendencias de performance
# - Análisis de capacidad
# - Optimizaciones sugeridas
# - Planificación de escalamiento
```

---

## 8. MANTENIMIENTO Y OPERACIONES

### 8.1 Tareas de Mantenimiento Rutinario

#### 8.1.1 Diario (Automatizado)
- Limpieza de logs temporales
- Verificación de espacio en disco
- Backup de configuraciones
- Verificación de health status

#### 8.1.2 Semanal
- Análisis de performance
- Rotación de logs
- Actualización de métricas
- Verificación de alertas

#### 8.1.3 Mensual
- Optimización de configuración
- Actualización de dependencias
- Revisión de capacidad
- Planificación de escalamiento

### 8.2 Mantenimiento Predictivo

#### 8.2.1 Análisis de Tendencias
```python
# Analizar tendencias de performance
def analyze_performance_trends(metrics_history):
    # CPU trends
    cpu_trend = analyze_cpu_trend(metrics_history)
    
    # Memory trends  
    memory_trend = analyze_memory_trend(metrics_history)
    
    # Convergence trends
    convergence_trend = analyze_convergence_trend(metrics_history)
    
    return {
        'cpu_trend': cpu_trend,
        'memory_trend': memory_trend,
        'convergence_trend': convergence_trend,
        'recommendations': generate_recommendations(cpu_trend, memory_trend)
    }
```

#### 8.2.2 Escalamiento Automático
```python
# Lógica de escalamiento
def should_scale_up(requirements_check, performance_summary):
    cpu_efficiency = performance_summary['resource_efficiency']['cpu_efficiency']
    memory_efficiency = performance_summary['resource_efficiency']['memory_efficiency']
    
    # Escalar si recursos subutilizados
    if cpu_efficiency < 50 and memory_efficiency < 50:
        return {
            'should_scale': True,
            'reason': 'Low resource utilization',
            'action': 'Increase workload or reduce resources'
        }
    
    # Escalar si sobrecarga
    if cpu_efficiency > 90 or memory_efficiency > 90:
        return {
            'should_scale': True,
            'reason': 'High resource utilization',
            'action': 'Add resources or reduce workload'
        }
    
    return {'should_scale': False}
```

---

## 9. TROUBLESHOOTING

### 9.1 Problemas Comunes

#### 9.1.1 Error: PySCF no encontrado
```bash
# Diagnóstico
python3 -c "import pyscf; print('PySCF OK')"

# Solución
pip install --force-reinstall pyscf==2.3.0 --no-binary=pyscf
```

#### 9.1.2 Error: Memoria insuficiente
```bash
# Verificar memoria
free -h

# Ajustar configuración
# Reducir max_workers o memory_limit_gb en config
```

#### 9.1.3 Error: Fallos de convergencia
```python
# Verificar parámetros de convergencia
# Ajustar conv_tol o sigma_ha en configuración
# Revisar calidad de pseudopotenciales
```

### 9.2 Diagnóstico Avanzado

#### 9.2.1 Análisis de Logs
```bash
# Buscar errores específicos
grep -i "error\|exception\|failed" logs/production.log

# Analizar alertas recientes
grep -i "alert\|warning" monitoring/alerts_*.log

# Verificar rendimiento
grep -i "duration\|time" logs/production.log
```

#### 9.2.2 Análisis de Métricas
```python
# Exportar y analizar métricas
monitor.export_metrics("diagnostic_export.json")

# Generar reporte de diagnóstico
def generate_diagnostic_report(pipeline):
    status = pipeline.get_monitoring_status()
    progress = pipeline.get_pipeline_progress()
    requirements = pipeline.get_system_requirements_check()
    
    return {
        'monitoring_status': status,
        'pipeline_progress': progress,
        'system_requirements': requirements,
        'recommendations': generate_troubleshooting_recommendations(
            status, progress, requirements
        )
    }
```

---

## 10. PROCEDIMIENTOS DE RECUPERACIÓN

### 10.1 Recovery Automático

#### 10.1.1 Detección y Corrección Automática
```python
# El sistema incluye auto-recovery para:
# - Fallos de convergencia
# - Timeouts de stage
# - Problemas de memoria
# - Errores de configuración

# Configuración de auto-recovery
AUTO_RECOVERY_CONFIG = {
    'enabled': True,
    'max_attempts': 3,
    'backoff_factor': 2.0,
    'strategies': [
        'reduce_workers',
        'increase_timeout', 
        'clear_memory',
        'restart_stage'
    ]
}
```

#### 10.1.2 Fallback a Configuración Segura
```python
# En caso de fallo repetido, usar configuración conservadora
SAFE_FALLBACK_CONFIG = {
    'max_workers': 2,
    'cutoff_list': [80, 120],  # Solo cutoffs básicos
    'kmesh_list': [(2,2,2), (4,4,4)],  # Solo k-mesh básicos
    'stage_timeout': 1800,  # Timeout extendido
    'memory_limit_gb': 8.0  # Límite reducido
}
```

### 10.2 Recovery Manual

#### 10.2.1 Rollback de Configuración
```bash
# Rollback a configuración anterior
python3 -c "
from src.utils.config_rollback import ConfigRollback
rollback = ConfigRollback('config')
rollback.rollback_config('production', '20251111_120000')
"
```

#### 10.2.2 Restore desde Backup
```bash
# Restore completo desde backup
tar -xzf backup_pre_deployment_20251111_120000.tar.gz
systemctl restart preconvergencia-gaas
```

### 10.3 Procedimientos de Emergencia

#### 10.3.1 Parada de Emergencia
```bash
# Parar todos los procesos
systemctl stop preconvergencia-gaas
pkill -f "preconvergencia"

# Liberar recursos
echo 3 > /proc/sys/vm/drop_caches

# Verificar estado
ps aux | grep python
```

#### 10.3.2 Recovery en Modo Seguro
```bash
# Iniciar en modo seguro (sin monitoreo)
python3 -c "
from config.settings import get_fast_config
from workflow.pipeline import PreconvergencePipeline

async def safe_mode():
    config = get_fast_config()
    pipeline = PreconvergencePipeline(config, enable_monitoring=False)
    result = await pipeline.execute()
    print(f'Safe mode result: {result.success}')

asyncio.run(safe_mode())
"
```

---

## 11. MÉTRICAS DE ÉXITO

### 11.1 KPIs de Producción

#### 11.1.1 Disponibilidad
- **Target**: 99.5% uptime
- **Métrica**: (Tiempo operativo / Tiempo total) * 100
- **Medición**: Continua

#### 11.1.2 Performance
- **Target**: < 2s tiempo de inicialización
- **Métrica**: Tiempo desde start hasta primer cálculo
- **Medición**: Por ejecución

#### 11.1.3 Eficiencia
- **Target**: < 200MB memoria base
- **Métrica**: Memoria sin cálculos activos
- **Medición**: Continua

#### 11.1.4 Calidad
- **Target**: > 90% tasa de convergencia
- **Métrica**: Cálculos convergidos / Total cálculos
- **Medición**: Por stage

### 11.2 Reportes de Performance

#### 11.2.1 Dashboard en Tiempo Real
```python
# Métricas mostradas en dashboard:
# - CPU usage por stage
# - Memory usage por calculation
# - Convergence rate por tipo
# - Alert frequency
# - System health score
```

#### 11.2.2 Reportes Periódicos
```python
# Reporte semanal incluye:
WEEKLY_REPORT = {
    'executive_summary': 'Overall system performance',
    'technical_details': 'Detailed metrics analysis',
    'alerts_summary': 'Alert patterns and resolutions',
    'recommendations': 'Optimization opportunities',
    'capacity_planning': 'Resource requirements forecast'
}
```

---

## 12. APÉNDICES

### 12.1 Comandos de Referencia Rápida

#### Validación
```bash
# Validación completa
./scripts/validate_production_environment.sh

# Test de componentes
python3 src/utils/environment_validator.py
python3 src/utils/production_monitor.py
```

#### Monitoreo
```bash
# Ver estado del sistema
systemctl status preconvergencia-gaas

# Ver logs en tiempo real
tail -f logs/production.log

# Ver métricas actuales
ls -la monitoring/
```

#### Mantenimiento
```bash
# Backup manual
tar -czf manual_backup_$(date +%Y%m%d).tar.gz src/ config/

# Limpieza de logs
find logs/ -name "*.log" -mtime +7 -delete

# Verificación de disco
df -h .
```

### 12.2 Configuraciones de Ejemplo

#### 12.2.1 Configuración de Desarrollo
```yaml
# config/development.yaml
development_config:
  cutoff_list: [80, 120]
  kmesh_list: [(2,2,2), (4,4,4)]
  max_workers: 2
  timeout_seconds: 300
  stage_timeout: 600
  memory_limit_gb: 4.0
  log_level: "DEBUG"
```

#### 12.2.2 Configuración HPC
```yaml
# config/hpc.yaml
hpc_config:
  cutoff_list: [60, 80, 100, 120, 140, 160, 180]
  kmesh_list: [(2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6)]
  max_workers: 32
  timeout_seconds: 3600
  stage_timeout: 7200
  memory_limit_gb: 64.0
  use_gpu: true
  gpu_memory_limit: 16.0
```

### 12.3 Contactos de Soporte

#### Escalación de Problemas
1. **Nivel 1**: Monitoreo automático y auto-recovery
2. **Nivel 2**: Administrador de sistema
3. **Nivel 3**: Equipo de desarrollo
4. **Nivel 4**: Consultor especializado

#### Información de Contacto
- **Emergencias 24/7**: [emergency@company.com]
- **Soporte técnico**: [support@company.com]
- **Desarrollo**: [dev@company.com]
- **Consultor**: [consultant@company.com]

---

**Documento generado automáticamente por el Sistema de Análisis de Despliegue v2.0**  
**Última actualización**: 2025-11-11  
**Próxima revisión**: 2025-12-11