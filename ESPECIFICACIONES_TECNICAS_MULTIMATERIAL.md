# ESPECIFICACIONES TÉCNICAS DETALLADAS
## Expansión Multi-Material para Preconvergencia DFT

### 1. DISEÑO DE BASE DE DATOS EXPANDIDA

#### 1.1 Esquema Principal de Datos

```python
# src/models/materials.py
@dataclass
class SemiconductorMaterial:
    """Base de datos de materiales semiconductores."""
    
    # Identificación
    formula: str
    name: str
    material_type: str  # "III-V", "II-VI", "IV-IV"
    crystal_structure: str  # "zincblende", "wurtzite", "diamond"
    
    # Composición atómica
    group_iii_element: Optional[str] = None  # Para III-V
    group_v_element: Optional[str] = None    # Para III-V
    group_ii_element: Optional[str] = None   # Para II-VI
    group_vi_element: Optional[str] = None   # Para II-VI
    
    # Propiedades atómicas
    atomic_properties: Dict[str, AtomicProperty] = field(default_factory=dict)
    
    # Propiedades de red
    lattice_constant: Optional[float] = None  # Å
    experimental_lattice: Optional[float] = None  # Å
    lattice_uncertainty: Optional[float] = None   # Å
    bond_length: Optional[float] = None     # Å
    
    # Propiedades electrónicas
    band_gap: Optional[float] = None      # eV
    experimental_bandgap: Optional[float] = None  # eV
    dielectric_constant: Optional[float] = None
    effective_mass_electron: Optional[float] = None  # m_e
    effective_mass_hole: Optional[float] = None     # m_e
    
    # Referencias experimentales
    reference_doi: Optional[str] = None
    reference_year: Optional[int] = None
    
    # Metadata
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_level: str = "high"  # "high", "medium", "low"

@dataclass
class AtomicProperty:
    """Propiedades de elementos individuales."""
    
    element: str
    atomic_number: int
    atomic_mass: float  # u
    covalent_radius: float  # Å
    ionic_radius: Optional[float] = None  # Å (estado de valencia común)
    electronegativity: float  # Pauling
    ionization_energy: float  # eV
    electron_affinity: float  # eV
    
    # Configuración electrónica
    electron_configuration: str
    
    # Pseudopotenciales disponibles
    available_pseudopotentials: List[str] = field(default_factory=list)
    
    # Notas de valencia química
    common_valence: int = 0
```

#### 1.2 Estructura de Directorios para Base de Datos

```
database/
├── elements/
│   ├── group_III.json      # Al, Ga, In, B, Tl
│   ├── group_V.json        # N, P, As, Sb, Bi
│   ├── group_II.json       # Be, Mg, Ca, Sr, Ba, Zn, Cd, Hg
│   └── group_VI.json       # O, S, Se, Te, Po
├── materials/
│   ├── III-V/
│   │   ├── GaAs.json
│   │   ├── GaN.json
│   │   ├── InAs.json
│   │   └── ...
│   └── II-VI/
│       ├── ZnS.json
│       ├── ZnSe.json
│       ├── CdTe.json
│       └── ...
├── crystal_structures/
│   ├── zincblende.json
│   ├── wurtzite.json
│   └── diamond.json
└── references/
    └── experimental_data.json
```

### 2. SISTEMA DE PERMUTACIONES POR GRUPO

#### 2.1 Grupos de Tabla Periódica Predefinidos

```python
# src/utils/periodic_groups.py
PERIODIC_GROUPS = {
    "group_III": {
        "elements": ["Al", "Ga", "In", "B", "Tl"],
        "properties": {
            "valence": 3,
            "typical_coordination": [4, 6],
            "oxidation_states": [1, 3]
        }
    },
    "group_V": {
        "elements": ["N", "P", "As", "Sb", "Bi"],
        "properties": {
            "valence": 5,
            "typical_coordination": [4],
            "oxidation_states": [-3, 3, 5]
        }
    },
    "group_II": {
        "elements": ["Be", "Mg", "Ca", "Sr", "Ba", "Zn", "Cd", "Hg"],
        "properties": {
            "valence": 2,
            "typical_coordination": [4, 6],
            "oxidation_states": [2]
        }
    },
    "group_VI": {
        "elements": ["O", "S", "Se", "Te", "Po"],
        "properties": {
            "valence": 6,
            "typical_coordination": [4, 6],
            "oxidation_states": [-2, 4, 6]
        }
    }
}
```

#### 2.2 Algoritmo de Generación de Permutaciones

```python
# src/core/permutation_generator.py
class PermutationGenerator:
    """Generador de combinaciones por grupos."""
    
    def __init__(self, database: MaterialDatabase):
        self.database = database
        
    def generate_iii_v_combinations(self) -> List[str]:
        """Genera todas las combinaciones III-V."""
        group_iii = PERIODIC_GROUPS["group_III"]["elements"]
        group_v = PERIODIC_GROUPS["group_V"]["elements"]
        
        combinations = []
        for element_iii in group_iii:
            for element_v in group_v:
                formula = f"{element_iii}{element_v}"
                if self._is_valid_combination(element_iii, element_v, "III-V"):
                    combinations.append(formula)
        return combinations
    
    def generate_ii_vi_combinations(self) -> List[str]:
        """Genera todas las combinaciones II-VI."""
        group_ii = PERIODIC_GROUPS["group_II"]["elements"]
        group_vi = PERIODIC_GROUPS["group_VI"]["elements"]
        
        combinations = []
        for element_ii in group_ii:
            for element_vi in group_vi:
                formula = f"{element_ii}{element_vi}"
                if self._is_valid_combination(element_ii, element_vi, "II-VI"):
                    combinations.append(formula)
        return combinations
```

### 3. ARQUITECTURA PIPELINE EXTENDIDA

#### 3.1 Configuración Multi-Material

```python
# src/config/multi_material_config.py
@dataclass
class MultiMaterialConfig(PreconvergenceConfig):
    """Configuración extendida para múltiples materiales."""
    
    # Lista de materiales a procesar
    materials_to_process: List[str] = field(default_factory=list)
    
    # Configuración específica por material
    material_configs: Dict[str, PreconvergenceConfig] = field(default_factory=dict)
    
    # Configuración de paralelismo
    parallel_materials: bool = True
    max_concurrent_materials: int = 4
    
    # Configuración de recursos
    resource_allocation: Dict[str, float] = field(default_factory=lambda: {
        "memory_per_material": 8.0,  # GB
        "cpu_per_material": 2.0      # cores
    })
    
    # Configuración de resultados
    results_organization: str = "by_material"  # "by_material", "by_property"
```

#### 3.2 Pipeline Multi-Material Principal

```python
# src/workflow/multi_material_pipeline.py
class MultiMaterialPipeline:
    """Pipeline extendido para múltiples materiales."""
    
    def __init__(self, config: MultiMaterialConfig):
        self.config = config
        self.database = MaterialDatabase(Path("database"))
        self.resource_manager = MultiMaterialResourceManager(config)
        self.monitoring_system = MultiMaterialMonitoringSystem()
        
    async def process_material_list(self, 
                                  materials: List[str],
                                  progress_callback: Optional[Callable] = None) -> MultiMaterialResults:
        """Procesa una lista de materiales."""
        results = {}
        
        # Organizar materiales por prioridad o recursos
        if self.config.parallel_materials:
            results = await self._process_parallel(materials, progress_callback)
        else:
            results = await self._process_sequential(materials, progress_callback)
        
        return MultiMaterialResults(results)
```

### 4. EJEMPLO DE CSV CON SEMICONDUCTORES II-VI

#### 4.1 Materiales Semiconductores (semiconductors_database.csv)

```
material_id,formula,name,material_type,group_iii,group_v,group_ii,group_vi,crystal_structure,lattice_constant_exp,lattice_constant_calc,bandgap_exp,bandgap_calc,bond_length,reference_doi,confidence_level
III-V_001,GaAs,Gallium Arsenide,III-V,Ga,As,,,"zincblende",5.653,5.647,1.42,1.39,2.45,10.1103/PhysRev.126.1691,high
III-V_002,GaP,Gallium Phosphide,III-V,Ga,P,,,"zincblende",5.450,5.448,2.26,2.28,2.36,10.1016/0038-1098(92)90005-F,high
III-V_003,GaN,Gallium Nitride,III-V,Ga,N,,,"wurtzite",3.189,3.190,3.40,3.25,1.95,10.1103/PhysRevB.61.15019,high
III-V_004,InAs,Indium Arsenide,III-V,In,As,,,"zincblende",6.058,6.058,0.36,0.35,2.61,10.1103/PhysRevB.64.115208,high
III-V_005,InP,Indium Phosphide,III-V,In,P,,,"zincblende",5.869,5.868,1.34,1.32,2.54,10.1016/0038-1098(92)90005-F,high
III-V_006,InN,Indium Nitride,III-V,In,N,,,"wurtzite",3.545,3.540,0.70,0.68,2.15,10.1103/PhysRevB.61.15019,high
III-V_007,AlAs,Aluminum Arsenide,III-V,Al,As,,,"zincblende",5.660,5.661,2.16,2.15,2.43,10.1103/PhysRevB.35.2557,high
III-V_008,AlP,Aluminum Phosphide,III-V,Al,P,,,"zincblende",5.463,5.463,2.45,2.50,2.37,10.1016/0038-1098(92)90005-F,high
III-V_009,InSb,Indium Antimonide,III-V,In,Sb,,,"zincblende",6.479,6.479,0.17,0.15,2.81,10.1103/PhysRevB.61.5002,high
II-VI_001,ZnS,Zinc Sulfide,II-VI,,,Zn,S,"zincblende",5.409,5.406,3.78,3.82,2.34,10.1103/PhysRevB.69.045203,high
II-VI_002,ZnSe,Zinc Selenide,II-VI,,,Zn,Se,"zincblende",5.667,5.667,2.70,2.72,2.45,10.1103/PhysRevB.69.045203,high
II-VI_003,ZnTe,Zinc Telluride,II-VI,,,Zn,Te,"zincblende",6.103,6.101,2.25,2.28,2.64,10.1103/PhysRevB.69.045203,high
II-VI_004,CdS,Cadmium Sulfide,II-VI,,,Cd,S,"wurtzite",4.136,4.135,2.42,2.45,2.52,10.1103/PhysRevB.65.035205,high
II-VI_005,CdSe,Cadmium Selenide,II-VI,,,Cd,Se,"wurtzite",4.299,4.298,1.74,1.76,2.62,10.1103/PhysRevB.65.035205,high
II-VI_006,CdTe,Cadmium Telluride,II-VI,,,Cd,Te,"zincblende",6.482,6.480,1.44,1.46,2.81,10.1103/PhysRevB.69.045203,high
II-VI_007,MgS,Magnesium Sulfide,II-VI,,,Mg,S,"rock_salt",5.200,5.198,4.50,4.45,2.40,10.1103/PhysRevB.70.205203,medium
II-VI_008,MgSe,Magnesium Selenide,II-VI,,,Mg,Se,"rock_salt",5.450,5.448,3.80,3.75,2.51,10.1103/PhysRevB.70.205203,medium
II-VI_009,BeO,Beryllium Oxide,II-VI,,,Be,O,"wurtzite",2.698,2.697,10.60,10.80,1.65,10.1103/PhysRevB.61.14067,medium
```

#### 4.2 Propiedades Atómicas (atomic_properties.csv)

```
element,symbol,atomic_number,atomic_mass,covalent_radius,ionic_radius_ii,ionic_radius_iii,ionic_radius_vi,electronegativity,ionization_energy,electron_configuration,group,common_valence,available_pseudopotentials
Aluminum,Al,13,26.982,1.18,0.535,,0.39,1.61,5.986,[Ne] 3s2 3p1,III,3,"gth-pbe,gth-pw91"
Gallium,Ga,31,69.723,1.22,0.620,,0.47,1.81,5.999,[Ar] 3d10 4s2 4p1,III,3,"gth-pbe,gth-pw91"
Indium,In,49,114.818,1.42,0.800,,0.62,1.78,5.786,[Kr] 4d10 5s2 5p1,III,3,"gth-pbe,gth-pw91"
Nitrogen,N,7,14.007,0.75,,,1.46,3.04,14.534,[He] 2s2 2p3,V,-3,"gth-pbe,gth-pw91"
Phosphorus,P,15,30.974,1.06,,,2.12,2.19,10.487,[Ne] 3s2 3p3,V,-3,"gth-pbe,gth-pw91"
Arsenic,As,33,74.922,1.19,,,2.22,2.18,9.789,[Ar] 3d10 4s2 4p3,V,-3,"gth-pbe,gth-pw91"
Zinc,Zn,30,65.380,1.20,0.740,,0.74,1.65,9.394,[Ar] 3d10 4s2,II,2,"gth-pbe,gth-pw91"
Cadmium,Cd,48,112.414,1.44,0.950,,0.95,1.69,8.994,[Kr] 4d10 5s2,II,2,"gth-pbe,gth-pw91"
Sulfur,S,16,32.065,1.02,,,2.19,2.58,10.360,[Ne] 3s2 3p4,VI,-2,"gth-pbe,gth-pw91"
Selenium,Se,34,78.971,1.16,,,2.38,2.55,9.752,[Ar] 3d10 4s2 4p4,VI,-2,"gth-pbe,gth-pw91"
Tellurium,Te,52,127.600,1.35,,,2.60,2.10,9.010,[Kr] 4d10 5s2 5p4,VI,-2,"gth-pbe,gth-pw91"
```

### 5. PLAN DE IMPLEMENTACIÓN DETALLADO

#### 5.1 Fases de Desarrollo

**FASE 1: Fundaciones (4-6 semanas)**
- Semana 1-2: Implementación de base de datos de materiales
- Semana 3-4: Sistema de permutaciones
- Semana 5-6: Integración con arquitectura existente

**FASE 2: Paralelización y Recursos (3-4 semanas)**
- Semana 7-8: Gestión de recursos paralelos
- Semana 9-10: Pipeline multi-material

**FASE 3: Monitoreo y Validación (3-4 semanas)**
- Semana 11-12: Sistema de monitoreo avanzado
- Semana 13-14: Validación y pruebas

#### 5.2 Tests de Integración

```python
# tests/integration/test_multi_material.py
class MultiMaterialIntegrationTests:
    """Tests de integración para sistema multi-material."""
    
    async def test_complete_workflow(self):
        """Test del flujo completo con 3 materiales."""
        materials = ["GaAs", "ZnS", "InP"]
        config = MultiMaterialConfig(materials_to_process=materials)
        
        pipeline = MultiMaterialPipeline(config)
        results = await pipeline.process_material_list(materials)
        
        # Verificar que todos los materiales se procesaron
        assert len(results.materials) == 3
        assert all(mat in results.materials for mat in materials)
        
        # Verificar calidad de resultados
        for material_result in results.materials.values():
            assert material_result.success
            assert material_result.total_duration > 0
```

### 6. CRITERIOS DE ÉXITO

#### 6.1 Criterios Técnicos
- **Cobertura de Materiales**: Al menos 15 materiales III-V y 10 materiales II-VI
- **Rendimiento**: Procesamiento paralelo con 4 materiales simultáneos sin degradación >20%
- **Precisión**: Parámetros de red calculados con error <1% vs experimental
- **Robustez**: Recuperación automática de fallos en ≥95% de casos

#### 6.2 Criterios de Usabilidad
- **Facilidad de Uso**: Configuración de nuevo material en ≤3 líneas de código
- **Documentación**: Documentación completa con ejemplos de uso
- **Extensibilidad**: Sistema permite añadir nuevos tipos de materiales

### 7. ENTREGABLES FINALES

1. ✅ **Diseño de base de datos expandida** - Completado en este documento
2. ✅ **Sistema de permutaciones** - Especificado con algoritmos detallados
3. ✅ **Arquitectura pipeline extendida** - Diseño multi-material completo
4. ✅ **CSV ejemplo funcional** - Datos de semiconductores II-VI específicos
5. ✅ **Plan de implementación detallado** - Roadmap de 12-14 semanas

### 8. PRÓXIMOS PASOS RECOMENDADOS

1. **Revisión Técnica**: Validación del diseño con el equipo científico
2. **Desarrollo MVP**: Implementación de funcionalidades core (Fase 1)
3. **Validación Inicial**: Tests con subset de materiales representativos
4. **Iteración**: Refinamiento basado en resultados iniciales
5. **Expansión Gradual**: Implementación de fases subsiguientes
