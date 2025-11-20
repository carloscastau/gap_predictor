# Sistema de Permutaciones Multimaterial para Semiconductores III-V y II-VI

## Descripción General

Este sistema implementa una arquitectura modular y escalable para generar, filtrar y configurar automáticamente combinaciones de semiconductores binarios III-V y II-VI. El sistema se integra perfectamente con el pipeline de preconvergencia existente y permite estudios sistemáticos de múltiples materiales.

## Arquitectura del Sistema

### Componentes Principales

```
src/
├── utils/
│   └── periodic_table_groups.py      # Base de datos de elementos
├── models/
│   └── semiconductor_database.py     # Base de datos de semiconductores
└── core/
    ├── material_permutator.py        # Generador de permutaciones
    └── multi_material_config.py      # Configuración multimaterial
```

## Módulos Implementados

### 1. Base de Datos de Elementos (`periodic_table_groups.py`)

**Propósito:** Proporciona información completa de elementos de la tabla periódica relevantes para semiconductores.

**Características:**
- ✅ Elementos de Grupo II: Be, Mg, Ca, Sr, Ba, Zn, Cd, Hg
- ✅ Elementos de Grupo III: B, Al, Ga, In, Tl
- ✅ Elementos de Grupo V: N, P, As, Sb, Bi
- ✅ Elementos de Grupo VI: O, S, Se, Te, Po

**Propiedades almacenadas:**
- Masa atómica (u)
- Radio iónico (Å)
- Electronegatividad (Pauling)
- Configuración electrónica
- Estados de oxidación
- Radio covalente (Å)
- Punto de fusión (K)
- Indicador de uso común en semiconductores

**Uso básico:**
```python
from src.utils.periodic_table_groups import get_element, PERIODIC_TABLE

# Obtener información de un elemento
ga = get_element('Ga')
print(f"Radio iónico: {ga.ionic_radius} Å")
print(f"Electronegatividad: {ga.electronegativity}")

# Listar elementos por grupo
group_iii = PERIODIC_TABLE.get_group_elements(PeriodicGroup.GROUP_III)
print(f"Elementos Grupo III: {list(group_iii.keys())}")
```

### 2. Base de Datos de Semiconductores (`semiconductor_database.py`)

**Propósito:** Gestiona información de semiconductores binarios conocidos y generados.

**Características:**
- ✅ 18+ semiconductores conocidos con propiedades experimentales
- ✅ Cálculo automático de propiedades derivadas
- ✅ Predicción de estructura cristalina
- ✅ Estimación de constante de red
- ✅ Validación de compatibilidad química

**Semiconductores III-V incluidos:**
- GaAs, GaN, InP, AlAs, InAs, GaP, InSb, AlN, InN, AlP, GaSb

**Semiconductores II-VI incluidos:**
- ZnS, ZnSe, ZnTe, CdS, CdSe, CdTe, HgTe

**Propiedades almacenadas:**
- Constante de red (Å)
- Band gap (eV)
- Estructura cristalina (Zincblende/Wurtzite/Rocksalt)
- Densidad (g/cm³)
- Movilidad electrónica/de huecos (cm²/V·s)
- Constante dieléctrica
- Conductividad térmica (W/m·K)

**Uso básico:**
```python
from src.models.semiconductor_database import SEMICONDUCTOR_DB

# Obtener semiconductor conocido
gaas = SEMICONDUCTOR_DB.get_semiconductor('GaAs')
print(f"Band gap: {gaas.properties.band_gap} eV")
print(f"Constante de red: {gaas.properties.lattice_constant} Å")

# Buscar por band gap
semiconductors = SEMICONDUCTOR_DB.search_by_band_gap(1.0, 2.0)
for sc in semiconductors:
    print(f"{sc.formula}: {sc.properties.band_gap} eV")

# Estadísticas
stats = SEMICONDUCTOR_DB.get_statistics()
print(f"Total: {stats['total']}, III-V: {stats['iii_v']}, II-VI: {stats['ii_vi']}")
```

### 3. Generador de Permutaciones (`material_permutator.py`)

**Propósito:** Genera automáticamente todas las combinaciones posibles de semiconductores con filtrado inteligente.

**Características:**
- ✅ Generación exhaustiva de combinaciones III-V y II-VI
- ✅ Filtrado por compatibilidad química
- ✅ Filtrado por radio iónico y electronegatividad
- ✅ Exclusión de elementos tóxicos/radiactivos (opcional)
- ✅ Filtros personalizados definidos por el usuario
- ✅ Búsqueda de materiales para heteroestructuras

**Filtros disponibles:**
```python
from src.core.material_permutator import PermutationFilter

filter_config = PermutationFilter(
    max_ionic_radius_ratio=2.5,        # Razón máxima de radios
    min_ionic_radius_ratio=0.2,        # Razón mínima de radios
    min_electronegativity_diff=0.3,    # Diferencia mínima EN
    max_electronegativity_diff=3.0,    # Diferencia máxima EN
    only_common_elements=True,         # Solo elementos comunes
    exclude_radioactive=True,          # Excluir radiactivos
    exclude_toxic=False                # Excluir tóxicos
)
```

**Uso básico:**
```python
from src.core.material_permutator import generate_all_iii_v, generate_all_ii_vi

# Generar todas las combinaciones III-V
result = generate_all_iii_v(filter_config)
print(f"Generadas: {result.total_generated}")
print(f"Aceptadas: {result.total_accepted}")
print(f"Tasa de aceptación: {result.acceptance_rate:.1f}%")

# Ver semiconductores aceptados
for sc in result.filtered_combinations:
    print(f"{sc.formula}: a≈{sc.estimate_lattice_constant():.3f}Å")
```

**Búsqueda de heteroestructuras:**
```python
from src.core.material_permutator import MATERIAL_PERMUTATOR

# Buscar materiales compatibles con GaAs
suggestions = MATERIAL_PERMUTATOR.suggest_heterostructures(
    base_material='GaAs',
    max_lattice_mismatch=0.02  # 2% máximo
)

for sc, mismatch in suggestions[:5]:
    print(f"{sc.formula}: mismatch={mismatch*100:.2f}%")
```

### 4. Configuración Multimaterial (`multi_material_config.py`)

**Propósito:** Gestiona configuraciones para simulaciones de múltiples materiales.

**Características:**
- ✅ Configuración base heredada por todos los materiales
- ✅ Parámetros específicos por material
- ✅ Generación automática desde permutaciones
- ✅ Priorización de materiales
- ✅ Ejecución paralela configurable
- ✅ Directorios de salida organizados
- ✅ Serialización YAML

**Uso básico:**
```python
from src.core.multi_material_config import (
    MultiMaterialConfig,
    create_common_semiconductors_config
)
from src.config.settings import get_fast_config

# Crear configuración con semiconductores comunes
config = create_common_semiconductors_config(
    base_config=get_fast_config()
)

# Ver estadísticas
stats = config.get_statistics()
print(f"Total materiales: {stats['total_materials']}")
print(f"III-V: {stats['iii_v_materials']}")
print(f"II-VI: {stats['ii_vi_materials']}")

# Agregar materiales específicos
config.add_material('GaAs', priority=10)
config.add_material('InP', priority=8)

# Guardar configuración
config.save_to_file('my_config.yaml')
```

**Configuración personalizada:**
```python
# Crear configuración vacía
config = MultiMaterialConfig(
    base_config=get_fast_config(),
    output_base_dir=Path("results_custom"),
    parallel_materials=True,
    max_concurrent_materials=4
)

# Agregar desde lista
materials = ['GaAs', 'InP', 'GaN', 'AlAs']
config.add_materials_from_list(materials)

# Agregar desde permutación
from src.core.material_permutator import generate_all_iii_v

result = generate_all_iii_v()
config.add_materials_from_permutation(result, max_materials=10)

# Obtener configuración específica para un material
gaas_config = config.get_material_config_dict('GaAs')
```

## Casos de Uso

### Caso 1: Generar Todas las Combinaciones III-V

```python
from src.core.material_permutator import generate_all_iii_v, PermutationFilter

# Configurar filtros
filter_config = PermutationFilter(
    only_common_elements=True,
    min_electronegativity_diff=0.4
)

# Generar combinaciones
result = generate_all_iii_v(filter_config)

# Analizar resultados
print(f"Resumen: {result}")
for sc in result.filtered_combinations:
    print(f"{sc.formula}:")
    print(f"  Razón radios: {sc.ionic_radius_ratio:.3f}")
    print(f"  Diff EN: {sc.electronegativity_difference:.3f}")
    print(f"  Estructura: {sc.predicted_crystal_structure.value}")
```

### Caso 2: Buscar Materiales para Heteroestructuras

```python
from src.core.material_permutator import MATERIAL_PERMUTATOR

# Buscar materiales compatibles con GaAs para heteroestructuras
base = 'GaAs'
compatible = MATERIAL_PERMUTATOR.suggest_heterostructures(
    base_material=base,
    max_lattice_mismatch=0.01  # 1% máximo
)

print(f"Materiales compatibles con {base}:")
for sc, mismatch in compatible:
    lattice = (sc.properties.lattice_constant 
              if sc.properties and sc.properties.lattice_constant
              else sc.estimate_lattice_constant())
    print(f"  {sc.formula}: a={lattice:.3f}Å, mismatch={mismatch*100:.2f}%")
```

### Caso 3: Configuración para Estudio Sistemático

```python
from src.core.multi_material_config import create_iii_v_config
from src.config.settings import get_production_config

# Crear configuración para todos los III-V
config = create_iii_v_config(
    base_config=get_production_config()
)

# Filtrar por band gap deseado
target_materials = []
for material in config.materials:
    if material.semiconductor and material.semiconductor.properties:
        bg = material.semiconductor.properties.band_gap
        if bg and 1.0 <= bg <= 2.0:
            target_materials.append(material)

# Crear nueva configuración con materiales filtrados
filtered_config = MultiMaterialConfig(
    base_config=get_production_config(),
    materials=target_materials
)

# Guardar
filtered_config.save_to_file('iii_v_bandgap_1_2eV.yaml')
```

### Caso 4: Permutaciones Personalizadas

```python
from src.core.material_permutator import MaterialPermutator, SemiconductorType

permutator = MaterialPermutator()

# Generar solo combinaciones de Ga e In con N, P, As
result = permutator.generate_custom_combinations(
    cations=['Ga', 'In'],
    anions=['N', 'P', 'As'],
    semiconductor_type=SemiconductorType.III_V
)

print(f"Combinaciones generadas: {result.total_accepted}")
for sc in result.filtered_combinations:
    print(f"{sc.formula}: a≈{sc.estimate_lattice_constant():.3f}Å")
```

## Integración con Pipeline Existente

El sistema se integra perfectamente con la arquitectura existente:

```python
from src.config.settings import PreconvergenceConfig
from src.core.multi_material_config import MultiMaterialConfig

# Configuración base del pipeline
base_config = PreconvergenceConfig(
    lattice_constant=5.653,
    cutoff_list=[80, 120, 160],
    kmesh_list=[(2,2,2), (4,4,4), (6,6,6)]
)

# Extender para múltiples materiales
multi_config = MultiMaterialConfig(base_config=base_config)
multi_config.add_materials_from_list(['GaAs', 'InP', 'GaN'])

# Cada material hereda la configuración base
# pero puede tener parámetros específicos
for material in multi_config.materials:
    material_config = multi_config.get_material_config_dict(material.formula)
    # Usar material_config con el pipeline existente
```

## Validación y Testing

Ejecutar el script de demostración:

```bash
cd /home/sorlac/Documentos/doctorate/preconvergencia-GaAs
PYTHONPATH=$PWD:$PYTHONPATH python examples/demo_multimaterial_system.py
```

El script demuestra:
1. ✅ Acceso a base de datos de elementos
2. ✅ Consulta de semiconductores conocidos
3. ✅ Generación de permutaciones III-V y II-VI
4. ✅ Búsqueda de heteroestructuras
5. ✅ Configuración multimaterial
6. ✅ Permutaciones personalizadas

## Estadísticas del Sistema

### Base de Datos de Elementos
- **Grupo II:** 8 elementos (Be, Mg, Ca, Sr, Ba, Zn, Cd, Hg)
- **Grupo III:** 5 elementos (B, Al, Ga, In, Tl)
- **Grupo V:** 5 elementos (N, P, As, Sb, Bi)
- **Grupo VI:** 5 elementos (O, S, Se, Te, Po)
- **Total:** 23 elementos

### Base de Datos de Semiconductores
- **III-V conocidos:** 11 semiconductores
- **II-VI conocidos:** 7 semiconductores
- **Total inicial:** 18 semiconductores con propiedades experimentales

### Capacidad de Generación
- **Combinaciones III-V posibles:** 5 × 5 = 25
- **Combinaciones II-VI posibles:** 8 × 5 = 40
- **Total teórico:** 65 semiconductores binarios
- **Aceptados con filtros por defecto:** ~20 semiconductores

## Extensibilidad

### Agregar Nuevos Elementos

```python
from src.utils.periodic_table_groups import ElementData, PeriodicGroup

new_element = ElementData(
    symbol='X',
    name='Nuevo Elemento',
    atomic_number=99,
    group=PeriodicGroup.GROUP_III,
    atomic_mass=100.0,
    ionic_radius=0.5,
    electronegativity=2.0,
    electron_config='[Xe] 6s² 6p¹',
    oxidation_states=[3],
    covalent_radius=1.2,
    common_in_semiconductors=True
)

PERIODIC_TABLE.group_iii['X'] = new_element
```

### Agregar Nuevos Semiconductores

```python
from src.models.semiconductor_database import (
    SemiconductorProperties,
    CrystalStructure,
    SEMICONDUCTOR_DB
)

new_props = SemiconductorProperties(
    formula='XY',
    lattice_constant=5.5,
    band_gap=1.5,
    crystal_structure=CrystalStructure.ZINCBLENDE,
    is_stable=True,
    is_experimental=False
)

# El semiconductor se agregará automáticamente al generar permutaciones
```

### Filtros Personalizados

```python
from src.core.material_permutator import PermutationFilter

def custom_filter(semiconductor):
    """Acepta solo semiconductores con band gap > 2.0 eV."""
    if semiconductor.properties and semiconductor.properties.band_gap:
        return semiconductor.properties.band_gap > 2.0
    return False

filter_config = PermutationFilter(
    custom_filters=[custom_filter]
)
```

## Mejores Prácticas

1. **Usar filtros apropiados:** Ajustar filtros según el objetivo del estudio
2. **Validar compatibilidad:** Verificar `is_chemically_compatible()` antes de simulaciones
3. **Priorizar materiales:** Asignar prioridades para optimizar recursos
4. **Guardar configuraciones:** Serializar configuraciones para reproducibilidad
5. **Verificar propiedades:** Usar propiedades experimentales cuando estén disponibles

## Limitaciones Conocidas

- ⚠️ Estimación de constante de red es aproximada para materiales sin datos experimentales
- ⚠️ Predicción de estructura cristalina usa reglas simplificadas de Pauling
- ⚠️ No incluye semiconductores ternarios o cuaternarios (futuro)
- ⚠️ Propiedades de algunos semiconductores menos comunes pueden estar incompletas

## Roadmap Futuro

- [ ] Soporte para semiconductores ternarios (AlGaAs, InGaAs, etc.)
- [ ] Soporte para semiconductores cuaternarios
- [ ] Predicción de band gap usando modelos ML
- [ ] Integración con bases de datos externas (Materials Project, AFLOW)
- [ ] Visualización de diagramas de fase
- [ ] Cálculo de desajuste de red en heteroestructuras multicapa

## Referencias

- Propiedades de semiconductores: Adachi, S. "Properties of Group-IV, III-V and II-VI Semiconductors"
- Estructuras cristalinas: Reglas de Pauling para compuestos iónicos
- Datos experimentales: Materials Project, NREL, literatura científica

## Soporte

Para reportar problemas o sugerir mejoras, contactar al equipo de desarrollo o crear un issue en el repositorio del proyecto.
