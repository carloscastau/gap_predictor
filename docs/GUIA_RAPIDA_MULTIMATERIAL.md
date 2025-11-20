# Guía Rápida: Sistema de Permutaciones Multimaterial

## Inicio Rápido (5 minutos)

### 1. Generar Todas las Combinaciones III-V

```python
from src.core.material_permutator import generate_all_iii_v

# Generar con filtros por defecto
result = generate_all_iii_v()

print(f"✓ Generadas: {result.total_generated} combinaciones")
print(f"✓ Aceptadas: {result.total_accepted} semiconductores")
print(f"✓ Tasa: {result.acceptance_rate:.1f}%")

# Ver primeros 5 semiconductores
for sc in result.filtered_combinations[:5]:
    print(f"  • {sc.formula}: a≈{sc.estimate_lattice_constant():.3f}Å")
```

### 2. Consultar Semiconductor Conocido

```python
from src.models.semiconductor_database import SEMICONDUCTOR_DB

# Obtener GaAs
gaas = SEMICONDUCTOR_DB.get_semiconductor('GaAs')

print(f"Material: {gaas.formula}")
print(f"Band gap: {gaas.properties.band_gap} eV")
print(f"Constante de red: {gaas.properties.lattice_constant} Å")
print(f"Movilidad e⁻: {gaas.properties.electron_mobility} cm²/V·s")
```

### 3. Buscar Materiales para Heteroestructuras

```python
from src.core.material_permutator import MATERIAL_PERMUTATOR

# Buscar compatibles con GaAs (±1% lattice mismatch)
compatible = MATERIAL_PERMUTATOR.suggest_heterostructures(
    base_material='GaAs',
    max_lattice_mismatch=0.01
)

print("Materiales compatibles con GaAs:")
for sc, mismatch in compatible[:5]:
    print(f"  • {sc.formula}: mismatch={mismatch*100:.2f}%")
```

### 4. Crear Configuración Multimaterial

```python
from src.core.multi_material_config import create_common_semiconductors_config
from src.config.settings import get_fast_config

# Crear configuración con semiconductores comunes
config = create_common_semiconductors_config(
    base_config=get_fast_config()
)

print(f"✓ Configurados {len(config)} materiales")

# Guardar configuración
config.save_to_file('mi_configuracion.yaml')
print("✓ Configuración guardada")
```

## Ejemplos Comunes

### Filtrar por Band Gap

```python
from src.models.semiconductor_database import SEMICONDUCTOR_DB

# Buscar semiconductores con band gap entre 1.0 y 2.0 eV
semiconductors = SEMICONDUCTOR_DB.search_by_band_gap(1.0, 2.0)

for sc in semiconductors:
    print(f"{sc.formula}: {sc.properties.band_gap} eV")
```

### Generar Solo Combinaciones Específicas

```python
from src.core.material_permutator import MaterialPermutator, SemiconductorType

permutator = MaterialPermutator()

# Solo combinaciones de Ga e In con N, P, As
result = permutator.generate_custom_combinations(
    cations=['Ga', 'In'],
    anions=['N', 'P', 'As'],
    semiconductor_type=SemiconductorType.III_V
)

print(f"Generadas {result.total_accepted} combinaciones")
```

### Filtros Personalizados

```python
from src.core.material_permutator import PermutationFilter, generate_all_iii_v

# Configurar filtros estrictos
strict_filter = PermutationFilter(
    max_ionic_radius_ratio=1.5,
    min_electronegativity_diff=0.5,
    only_common_elements=True,
    exclude_toxic=True
)

result = generate_all_iii_v(strict_filter)
print(f"Con filtros estrictos: {result.total_accepted} aceptados")
```

## Ejecutar Demo Completa

```bash
cd /home/sorlac/Documentos/doctorate/preconvergencia-GaAs
PYTHONPATH=$PWD:$PYTHONPATH python examples/demo_multimaterial_system.py
```

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| [`src/utils/periodic_table_groups.py`](../src/utils/periodic_table_groups.py) | Base de datos de elementos |
| [`src/models/semiconductor_database.py`](../src/models/semiconductor_database.py) | Base de datos de semiconductores |
| [`src/core/material_permutator.py`](../src/core/material_permutator.py) | Generador de permutaciones |
| [`src/core/multi_material_config.py`](../src/core/multi_material_config.py) | Configuración multimaterial |
| [`examples/demo_multimaterial_system.py`](../examples/demo_multimaterial_system.py) | Script de demostración |

## Documentación Completa

Ver [`docs/SISTEMA_MULTIMATERIAL.md`](SISTEMA_MULTIMATERIAL.md) para documentación detallada.

## Estadísticas del Sistema

- **Elementos disponibles:** 23 (Grupos II, III, V, VI)
- **Semiconductores conocidos:** 18 (11 III-V, 7 II-VI)
- **Combinaciones posibles:** 65 (25 III-V, 40 II-VI)
- **Propiedades por elemento:** 11 (masa, radio, electronegatividad, etc.)
- **Propiedades por semiconductor:** 13 (lattice, band gap, movilidad, etc.)

## Soporte

Para más información, consultar la documentación completa o ejecutar el script de demostración.
