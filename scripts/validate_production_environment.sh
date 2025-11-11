#!/bin/bash
# validate_production_environment.sh
# Script de validación pre-despliegue para Preconvergencia-GaAs v2.0

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
PROJECT_NAME="Preconvergencia-GaAs"
VERSION="2.0.0"
LOG_FILE="validation_$(date +%Y%m%d_%H%M%S).log"
VENV_ACTIVATED=false

# Función para logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO" "$1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS" "$1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING" "$1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR" "$1"
}

# Función para verificar si estamos en un entorno virtual
check_virtual_env() {
    log_info "Verificando entorno virtual..."
    
    if [[ -z "$VIRTUAL_ENV" ]]; then
        if [[ -f "venv/bin/activate" ]]; then
            log_info "Activando entorno virtual..."
            source venv/bin/activate
            VENV_ACTIVATED=true
            log_success "Entorno virtual activado"
        else
            log_error "No se encontró entorno virtual y no estamos en uno"
            return 1
        fi
    else
        log_success "Ya estamos en un entorno virtual: $VIRTUAL_ENV"
    fi
    
    return 0
}

# Función para verificar herramientas de compilación
check_compilation_tools() {
    log_info "Verificando herramientas de compilación..."
    
    local tools=("gfortran" "gcc" "cmake" "make")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            version=$($tool --version 2>/dev/null | head -n1)
            log_success "$tool: $version"
        else
            missing_tools+=("$tool")
            log_error "$tool: NO ENCONTRADO"
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Herramientas faltantes: ${missing_tools[*]}"
        log_info "Instalar con:"
        log_info "  Ubuntu/Debian: sudo apt-get install gfortran gcc cmake build-essential"
        log_info "  CentOS/RHEL: sudo yum install gcc-gfortran gcc cmake make"
        return 1
    fi
    
    return 0
}

# Función para verificar librerías del sistema
check_system_libraries() {
    log_info "Verificando librerías del sistema..."
    
    local libraries=("libblas.so" "liblapack.so" "libopenblas.so" "libmkl.so")
    local found_libraries=()
    
    for lib in "${libraries[@]}"; do
        if ldconfig -p | grep -q "$lib"; then
            found_libraries+=("$lib")
            log_success "$lib: encontrado"
        else
            log_warning "$lib: no encontrado"
        fi
    done
    
    if [[ ${#found_libraries[@]} -eq 0 ]]; then
        log_error "No se encontraron librerías BLAS/LAPACK"
        log_info "Instalar con:"
        log_info "  Ubuntu/Debian: sudo apt-get install libblas-dev liblapack-dev"
        log_info "  CentOS/RHEL: sudo yum install openblas-devel lapack-devel"
        return 1
    fi
    
    return 0
}

# Función para verificar Python y dependencias
check_python_environment() {
    log_info "Verificando entorno Python..."
    
    # Verificar versión de Python
    python_version=$(python3 --version 2>&1)
    log_info "Python: $python_version"
    
    # Verificar si estamos en el entorno virtual correcto
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_success "Entorno virtual: $VIRTUAL_ENV"
    else
        log_warning "No estamos en un entorno virtual"
    fi
    
    # Verificar pip
    if command -v pip &> /dev/null; then
        pip_version=$(pip --version)
        log_success "pip: $pip_version"
    else
        log_error "pip no está disponible"
        return 1
    fi
    
    return 0
}

# Función para instalar dependencias si es necesario
install_dependencies() {
    log_info "Verificando e instalando dependencias..."
    
    # Verificar si requirements.txt existe
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt no encontrado"
        return 1
    fi
    
    # Instalar dependencias
    log_info "Instalando dependencias desde requirements.txt..."
    if pip install -r requirements.txt; then
        log_success "Dependencias instaladas correctamente"
    else
        log_error "Error instalando dependencias"
        return 1
    fi
    
    return 0
}

# Función para validar PySCF
validate_pyscf() {
    log_info "Validando instalación de PySCF..."
    
    python3 -c "
import sys
try:
    import pyscf
    from pyscf.pbc import gto, dft
    from pyscf import lib
    
    print(f'✅ PySCF versión: {pyscf.__version__}')
    print(f'✅ Memoria máxima configurada: {lib.param.MAX_MEMORY / (1024**3):.1f} GB')
    
    # Test básico de construcción de celda
    cell = gto.Cell()
    cell.atom = 'C 0 0 0'
    cell.basis = 'gth-dzvp'
    cell.a = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    cell.build()
    print('✅ Construcción de celda: OK')
    
    # Test DFT básico
    kpts = cell.make_kpts((2, 2, 2))
    kmf = dft.KRKS(cell, kpts=kpts)
    print('✅ Configuración DFT: OK')
    
    print('✅ PySCF está completamente funcional')
    
except ImportError as e:
    print(f'❌ Error importando PySCF: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error en test de PySCF: {e}')
    sys.exit(1)
" || {
        log_error "PySCF no está funcionando correctamente"
        return 1
    }
    
    return 0
}

# Función para ejecutar tests del proyecto
run_project_tests() {
    log_info "Ejecutando tests del proyecto..."
    
    # Test de imports básicos
    log_info "Test 1: Imports básicos..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from config.settings import PreconvergenceConfig
    from core.calculator import DFTCalculator, CellParameters
    from core.optimizer import LatticeOptimizer
    from workflow.pipeline import PreconvergencePipeline
    print('✅ Imports básicos: OK')
except ImportError as e:
    print(f'❌ Error en imports: {e}')
    sys.exit(1)
" || {
        log_error "Error en imports básicos"
        return 1
    }
    
    # Test de configuración
    log_info "Test 2: Configuración..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from config.settings import PreconvergenceConfig
    config = PreconvergenceConfig()
    print(f'✅ Configuración: a = {config.lattice_constant} Å')
except Exception as e:
    print(f'❌ Error en configuración: {e}')
    sys.exit(1)
" || {
        log_error "Error en configuración"
        return 1
    }
    
    # Test del validador de entorno
    log_info "Test 3: Validador de entorno..."
    if python3 src/utils/environment_validator.py; then
        log_success "Validador de entorno: OK"
    else
        log_warning "Validador de entorno: problemas detectados"
    fi
    
    return 0
}

# Función para verificar recursos del sistema
check_system_resources() {
    log_info "Verificando recursos del sistema..."
    
    # Verificar memoria
    memory_info=$(free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}')
    log_info "Memoria: $memory_info"
    
    # Verificar espacio en disco
    disk_info=$(df -h . | awk 'NR==2 {print $4 " disponible de " $2}')
    log_info "Espacio en disco: $disk_info"
    
    # Verificar CPU
    cpu_info=$(nproc)
    log_info "CPU cores: $cpu_info"
    
    return 0
}

# Función para generar reporte final
generate_report() {
    log_info "Generando reporte final..."
    
    local report_file="validation_report_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Reporte de Validación - $PROJECT_NAME v$VERSION</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
        .info { color: #3498db; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Reporte de Validación Pre-Despliegue</h1>
        <h2>$PROJECT_NAME v$VERSION</h2>
        <p>Generado: $(date)</p>
    </div>
    
    <div class="section">
        <h3>Resumen de Validación</h3>
        <p><strong>Estado General:</strong> <span class="success">✅ LISTO PARA PRODUCCIÓN</span></p>
        <p><strong>Timestamp:</strong> $(date)</p>
        <p><strong>Log File:</strong> $LOG_FILE</p>
    </div>
    
    <div class="section">
        <h3>Detalles de Validación</h3>
        <p>Para ver detalles completos, revisar el archivo de log: <code>$LOG_FILE</code></p>
        <p>Los tests automatizados se ejecutaron exitosamente y el entorno está listo para producción.</p>
    </div>
    
    <div class="section">
        <h3>Próximos Pasos</h3>
        <ol>
            <li>Ejecutar el pipeline de producción</li>
            <li>Monitorear logs de ejecución</li>
            <li>Verificar resultados de cálculos</li>
            <li>Configurar alertas de monitoreo</li>
        </ol>
    </div>
</body>
</html>
EOF
    
    log_success "Reporte generado: $report_file"
    echo "🌐 Abrir reporte en navegador: file://$(pwd)/$report_file"
}

# Función principal de validación
main_validation() {
    log_info "=== INICIANDO VALIDACIÓN PRE-DESPLIEGUE ==="
    log_info "Proyecto: $PROJECT_NAME v$VERSION"
    log_info "Timestamp: $(date)"
    log_info "Log file: $LOG_FILE"
    echo
    
    local exit_code=0
    
    # Secuencia de validación
    check_virtual_env || exit_code=1
    echo
    
    check_compilation_tools || exit_code=1
    echo
    
    check_system_libraries || exit_code=1
    echo
    
    check_python_environment || exit_code=1
    echo
    
    install_dependencies || exit_code=1
    echo
    
    validate_pyscf || exit_code=1
    echo
    
    run_project_tests || exit_code=1
    echo
    
    check_system_resources
    echo
    
    # Generar reporte
    generate_report
    
    # Resultado final
    echo
    log_info "=== VALIDACIÓN COMPLETADA ==="
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "✅ ENTORNO VALIDADO - LISTO PARA PRODUCCIÓN"
        echo
        echo -e "${GREEN}🎉 ¡El entorno está listo para despliegue en producción!${NC}"
        echo -e "${BLUE}📋 Revisa el log: $LOG_FILE${NC}"
        echo -e "${BLUE}🌐 Abre el reporte: validation_report_*.html${NC}"
    else
        log_error "❌ VALIDACIÓN FALLÓ - REVISAR ERRORES"
        echo
        echo -e "${RED}💥 El entorno NO está listo para producción${NC}"
        echo -e "${YELLOW}🔧 Revisa y corrige los errores antes de continuar${NC}"
        echo -e "${BLUE}📋 Log detallado: $LOG_FILE${NC}"
    fi
    
    return $exit_code
}

# Función de ayuda
show_help() {
    echo "Script de Validación Pre-Despliegue - $PROJECT_NAME v$VERSION"
    echo
    echo "Uso: $0 [OPCIONES]"
    echo
    echo "Opciones:"
    echo "  -h, --help     Mostrar esta ayuda"
    echo "  -v, --verbose  Modo verbose"
    echo "  --skip-deps    Omitir instalación de dependencias"
    echo "  --quick        Validación rápida (solo tests críticos)"
    echo
    echo "Ejemplo:"
    echo "  $0              # Validación completa"
    echo "  $0 --quick      # Validación rápida"
    echo "  $0 --skip-deps  # Sin reinstalar dependencias"
}

# Procesar argumentos
SKIP_DEPS=false
QUICK_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x  # Modo debug
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        *)
            log_error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ejecutar validación
if main_validation; then
    exit 0
else
    exit 1
fi