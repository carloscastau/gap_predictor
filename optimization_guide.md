# Guía de Optimización para Cálculos DFT de GaAs

## Problemas Identificados y Soluciones

### 1. Problema: "Moléculas en el mismo lugar"
**Causa probable:** Parámetro `x_ga` incorrecto o definición errónea de la celda unitaria.

**Solución implementada:**
- ✅ Verificación automática de distancia mínima Ga-As (> 1.0 Å)
- ✅ Validación de parámetros estructurales al inicio
- ✅ Corrección en la definición de la matriz de celda unitaria

### 2. Problema: Estructura de bandas incorrecta
**Causa probable:** Convergencia insuficiente o parámetros de smearing inadecuados.

**Soluciones implementadas:**
- ✅ Logging detallado del análisis de bandas (VBM/CBM)
- ✅ Verificación automática de número de bandas de valencia/conducción
- ✅ Detección de gaps sospechosamente pequeños (< 0.1 eV)

### 3. Problema: Estimación de 2π/A
**Causa probable:** Problemas en la generación o distribución de k-points.

**Soluciones implementadas:**
- ✅ Información detallada sobre distribución de k-points
- ✅ Cálculo de distancias Δk mínimas y máximas
- ✅ Verificación de generación correcta de mallas de k-points

## Parámetros Óptimos Recomendados

### Para cálculos de producción:
```bash
python preconvergencia_GaAs.py \
  --cutoff_list 80,120,160,200 \
  --k_list 6x6x6,8x8x8,10x10x10,12x12x12 \
  --a0 5.653 --da 0.02 --npoints_side 4 \
  --sigma_ha 0.01 --xc PBE \
  --basis_list gth-tzv2p,gth-tzvp \
  --timeout_s 600 --dos on --make_report on
```

### Para convergencia rápida (desarrollo):
```bash
python preconvergencia_GaAs.py --fast \
  --cutoff_list 80,120,160 \
  --k_list 4x4x4,6x6x6,8x8x8 \
  --npoints_side 3 --timeout_s 300
```

#### Para optimización avanzada con búsqueda global:
```bash
python preconvergencia_GaAs.py \
  --cutoff_list 100,150,200 \
  --k_list 8x8x8,10x10x10 \
  --a0 5.653 --da 0.03 --npoints_side 5 \
  --sigma_ha 0.01 --xc PBE \
  --basis_list gth-tzv2p \
  --timeout_s 900 --dos on --make_report on
```

## Pipeline Optimizado para Supercomputo

### Estrategia de Paralelización:

1. **Nivel 1:** Paralelización por bases (diferentes nodos)
2. **Nivel 2:** Paralelización por puntos de energía (mismo nodo)
3. **Nivel 3:** Paralelización por k-points (PySCF interno)

### Nueva Optimización Avanzada de Geometrías:

El proyecto ahora incluye técnicas avanzadas de optimización global:

#### **Multi-Start Optimization:**
- ✅ Exploración inicial amplia del espacio de parámetros
- ✅ Múltiples reinicios aleatorios con rango adaptativo
- ✅ Estrategia de escape de mínimos locales
- ✅ Detección automática de múltiples candidatos a mínimos

#### **Fases de Optimización:**
1. **Fase 1:** Exploración sistemática amplia
2. **Fase 2:** Multi-start con puntos aleatorios
3. **Fase 3:** Refinamiento local alrededor de candidatos
4. **Fase 4:** Análisis estadístico y detección de mínimos globales

#### **Características Avanzadas:**
- ✅ Detección automática de cambios de pendiente
- ✅ Análisis estadístico completo de resultados
- ✅ Visualización avanzada con múltiples paneles
- ✅ Resumen detallado de la optimización

### Variables de entorno recomendadas:
```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Validación de Resultados

### Criterios de convergencia:
- **Cutoff de energía:** ΔE < 1 meV entre puntos consecutivos
- **K-mesh:** ΔE < 1 meV entre mallas consecutivas
- **Parámetro de red:** |a_calculado - a0| < 0.02 Å

### Valores de referencia (GaAs):
- **Gap experimental:** 1.42 eV (300K)
- **Parámetro de red:** 5.653 Å
- **Tipo de gap:** Directo (Γ → Γ)

## Solución de Problemas Comunes

### Si el SCF no converge:
1. Aumentar `max_cycle` (por defecto: 80)
2. Reducir `conv_tol` gradualmente (por defecto: 1e-8)
3. Usar level shifting (`kmf.level_shift = 0.1`)

### Si las bandas parecen incorrectas:
1. Verificar parámetros estructurales con `--validate`
2. Aumentar densidad de k-points para bandas
3. Revisar smearing (`sigma_ha` entre 0.01-0.05 Ha)

### Si los tiempos son excesivos:
1. Usar `--fast` para desarrollo
2. Implementar timeouts más estrictos
3. Considerar reducción selectiva de parámetros