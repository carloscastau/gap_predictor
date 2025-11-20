# Documentación de la Base de Datos de Semiconductores II-VI

## Descripción General

Este documento describe la estructura y contenido de la base de datos de semiconductores II-VI, específicamente diseñada para análisis de propiedades físicas, electrónicas y ópticas de materiales semiconductores binarios.

**Archivo:** `data/semiconductores_ii_vi_ejemplo.csv`  
**Fecha de creación:** 2025-11-20  
**Versión:** 1.0  
**Autor:** Sistema de Análisis de Materiales  

## Propósito

La base de datos está enfocada en proporcionar datos experimentales validados para los semiconductores II-VI más relevantes: ZnS, ZnSe, ZnTe, CdS, CdSe y CdTe, incluyendo sus diferentes fases cristalinas, para facilitar:

- Análisis de tendencias periódicas
- Modelado de propiedades materiales
- Comparación con cálculos teóricos
- Desarrollo de modelos de aprendizaje automático
- Diseño de heterostructuras

## Estructura de la Base de Datos

### Identificación del Material

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `formula` | Fórmula química del compuesto | - | String |
| `grupo_cristalino` | Grupo cristalográfico del material | - | String |
| `estructura_cristalina` | Tipo de estructura cristalina | - | String |

#### Valores Posibles:
- **grupo_cristalino:** zincblende, wurtzite, rock_salt
- **estructura_cristalina:** cubic, hexagonal, orthorhombic

### Propiedades Atómicas

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `elemento_A` | Elemento del grupo II | - | String |
| `elemento_B` | Elemento del grupo VI | - | String |
| `numero_atomico_A` | Número atómico del elemento A | - | Integer |
| `numero_atomico_B` | Número atómico del elemento B | - | Integer |

### Propiedades Físicas Básicas

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `masa_molar` | Masa molar del compuesto | g/mol | Float |
| `g_cm3` | Densidad del material | g/cm³ | Float |
| `punto_fusion_K` | Punto de fusión | Kelvin | Float |
| `conductividad_termica_W_mK` | Conductividad térmica | W/m·K | Float |

### Propiedades de Red Cristalina

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `constante_red_a_angstrom` | Parámetro de red 'a' | Å | Float |
| `constante_red_c_angstrom` | Parámetro de red 'c' | Å | Float |
| `volumen_celda_angstrom3` | Volumen de la celda unitaria | Å³ | Float |
| `parametros_red_cubica` | Parámetros de red para estructuras cúbicas | Å | String |

**Nota:** `constante_red_c_angstrom` es NULL para estructuras cúbicas.

### Propiedades Electrónicas

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `band_gap_directo_eV` | Band gap directo | eV | Float |
| `band_gap_indirecto_eV` | Band gap indirecto | eV | Float |
| `movilidad_electrones_cm2_Vs` | Movilidad de electrones | cm²/V·s | Float |
| `movilidad_huecos_cm2_Vs` | Movilidad de huecos | cm²/V·s | Float |

### Propiedades Ópticas

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `indice_refraccion` | Índice de refracción | - | Float |
| `permitividad_estatica` | Permitividad estática (ε₀) | - | Float |
| `energia_exciton_eV` | Energía de excitón | eV | Float |

### Referencias y Condiciones Experimentales

| Columna | Descripción | Unidades | Tipo de Dato |
|---------|-------------|----------|--------------|
| `referencia_experimental` | Referencia bibliográfica | - | String |
| `doi` | DOI del artículo | - | String |
| `temperatura_medicion_K` | Temperatura de medición | Kelvin | Integer |

## Materiales Incluidos

### Materiales Principales (Especificados)

1. **ZnS (Sulfuro de Zinc)**
   - Estructuras: zincblende (cúbica) y wurtzite (hexagonal)
   - Band gap: ~3.72 eV (directo)

2. **ZnSe (Seleniuro de Zinc)**
   - Estructura: zincblende (cúbica)
   - Band gap: ~2.70 eV (directo)

3. **ZnTe (Telururo de Zinc)**
   - Estructura: zincblende (cúbica)
   - Band gap: ~2.26 eV (directo)

4. **CdS (Sulfuro de Cadmio)**
   - Estructura: wurtzite (hexagonal)
   - Band gap: ~2.42 eV (directo)

5. **CdSe (Seleniuro de Cadmio)**
   - Estructura: wurtzite (hexagonal)
   - Band gap: ~1.74 eV (directo)

6. **CdTe (Telururo de Cadmio)**
   - Estructura: zincblende (cúbica)
   - Band gap: ~1.44 eV (directo)

### Materiales de Contexto

Para proporcionar un contexto completo de la familia II-VI, se incluyen también:

- **HgS:** Semimetal con band gap negativo
- **HgTe:** Semiconductor con banda de conducción invertida
- **BeO:** Material con banda gap muy amplio

## Fuentes de Datos

Los datos experimentales han sido extraídos de la literatura científica revisada por pares, incluyendo:

### Referencias Principales

1. **Hummer, K. (1973)** - Propiedades ópticas de ZnS
   - DOI: 10.1103/PhysRevB.7.5202

2. **Tutihasi, S. (1967)** - Propiedades de ZnSe y MgTe
   - DOI: 10.1103/PhysRev.158.623

3. **Shen, H. (1991)** - Propiedades de CdS
   - DOI: 10.1016/0921-5107(91)90003-Z

4. **Nakamura, K. (1992)** - Propiedades de CdSe
   - DOI: 10.1016/0040-6090(92)90107-1

5. **Adachi, S. (1999)** - Propiedades de CdTe
   - DOI: 10.1016/S0925-3467(99)00011-7

## Validación de Datos

### Criterios de Validación

1. **Consistencia con tendencias periódicas:**
   - Los band gaps disminuyen del S al Te para el mismo catión
   - Las constantes de red aumentan del S al Te
   - Las densidades siguen tendencias periódicas esperadas

2. **Compatibilidad con valores de literatura:**
   - Los valores están dentro del rango esperado (±10% de valores estándar)
   - Las referencias bibliográficas son reconocidas

3. **Coherencia interna:**
   - Las estructuras cristalinas son las más estables a temperatura ambiente
   - Los parámetros de red son físicamente consistentes

### Ejemplo de Validación

```python
# Verificar tendencia ZnS > ZnSe > ZnTe
zn_chalcogenides = ['ZnS', 'ZnSe', 'ZnTe']
band_gaps = [3.72, 2.70, 2.26]
assert all(band_gaps[i] > band_gaps[i+1] for i in range(len(band_gaps)-1))
```

## Uso de la Base de Datos

### Carga con Pandas

```python
import pandas as pd

# Cargar la base de datos
df = pd.read_csv('data/semiconductores_ii_vi_ejemplo.csv')

# Filtrar materiales principales
materiales_principales = df[df['formula'].isin(['ZnS', 'ZnSe', 'ZnTe', 'CdS', 'CdSe', 'CdTe'])]

# Análisis de tendencias
print(df.groupby('elemento_A')['band_gap_directo_eV'].mean())
```

### Ejemplos de Análisis

```python
# 1. Tendencia periódica S > Se > Te
for elemento in ['Zn', 'Cd']:
    subset = df[df['elemento_A'] == elemento]
    print(f"{elemento} chalcogenides: {subset['band_gap_directo_eV'].values}")

# 2. Comparación estructura: zincblende vs wurtzite
zn_s_data = df[df['formula'] == 'ZnS']
print(zn_s_data[['grupo_cristalino', 'band_gap_directo_eV']])

# 3. Propiedades vs tamaño atómico
import matplotlib.pyplot as plt
plt.scatter(df['numero_atomico_B'], df['band_gap_directo_eV'])
plt.xlabel('Número atómico del anión')
plt.ylabel('Band gap (eV)')
plt.title('Tendencia periódica de band gap')
```

## Actualización y Mantenimiento

### Script de Generación

El archivo `scripts/generate_semiconductor_database.py` permite:

- Regenerar la base de datos con nuevos datos
- Validar datos existentes
- Generar estadísticas de la base de datos
- Incluir/excluir materiales de contexto

### Comandos de Uso

```bash
# Generar base de datos completa
python scripts/generate_semiconductor_database.py

# Solo materiales principales (sin contexto)
python scripts/generate_semiconductor_database.py --sin-contexto

# Validar datos existentes
python scripts/generate_semiconductor_database.py --validar

# Generar estadísticas
python scripts/generate_semiconductor_database.py --estadisticas
```

## Limitaciones y Consideraciones

1. **Temperatura:** Todos los datos están reportados a ~298K (temperatura ambiente)
2. **Pureza:** Los valores pueden variar con la pureza del cristal
3. **Orientación:** Para cristales no cúbicos, pueden existir anisotropías
4. **Actualización:** La base de datos debe actualizarse periódicamente con nuevos datos experimentales

## Formato de Salida

- **Encoding:** UTF-8
- **Delimiter:** Coma (,)
- **Missing values:** Representados como NULL o vacíos
- **Header:** Primera línea contiene nombres de columnas

## Extensiones Futuras

Posibles adiciones a la base de datos:

1. Propiedades térmicas adicionales (calor específico, expansión térmica)
2. Propiedades de defectos y dopaje
3. Propiedades bajo presión hidrostática
4. Datos de alta temperatura
5. Propiedades magnéticas para materiales dopados

---

**Contacto:** Para actualizaciones o correcciones, consulte el repositorio del proyecto o contacte al equipo de desarrollo.