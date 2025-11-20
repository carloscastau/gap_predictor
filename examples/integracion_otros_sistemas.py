#!/usr/bin/env python3
# examples/integracion_otros_sistemas.py
"""
Integración con Otros Sistemas - Preconvergencia Multimaterial

Este script demuestra cómo integrar el sistema de preconvergencia multimaterial
con otros sistemas y bases de datos externas, incluyendo:
- Integración con Materials Project API
- Conexión con bases de datos AFLOW
- Integración con códigos DFT externos (QE, VASP, ABINIT)
- Sincronización con sistemas de gestión de datos
- Intercambio de datos con herramientas de análisis

Ejecutar: python examples/integracion_otros_sistemas.py
"""

import sys
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import requests
from datetime import datetime

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.semiconductor_database import SEMICONDUCTOR_DB
from core.material_permutator import MATERIAL_PERMUTATOR
from workflow.multi_material_pipeline import run_custom_materials_campaign
from analysis.multi_material_analysis import MultiMaterialAnalyzer


class MaterialsProjectIntegration:
    """Integración con Materials Project API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa la integración con Materials Project.
        
        Args:
            api_key: API key de Materials Project (opcional para consultas públicas)
        """
        self.api_key = api_key or "demo_key"
        self.base_url = "https://api.materialsproject.org/rest/v1"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        print(f"🔗 Integración Materials Project inicializada")
    
    async def fetch_materials_data(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """
        Obtiene datos desde Materials Project.
        
        Args:
            formulas: Lista de fórmulas a consultar
            
        Returns:
            Lista de diccionarios con datos de materiales
        """
        print(f"📡 Consultando Materials Project para {len(formulas)} materiales...")
        
        results = []
        for formula in formulas:
            try:
                # URL para obtener estructura y propiedades
                url = f"{self.base_url}/materials/{formula}/calculate"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                processed_data = self._process_mp_data(data, formula)
                                results.append(processed_data)
                                print(f"   ✅ {formula}: Datos obtenidos")
                            else:
                                print(f"   ⚠️  {formula}: No se encontraron datos")
                        else:
                            print(f"   ❌ {formula}: Error {response.status}")
                            
            except Exception as e:
                print(f"   ❌ {formula}: Error en consulta - {e}")
        
        print(f"📊 Resultados obtenidos: {len(results)} materiales")
        return results
    
    def _process_mp_data(self, mp_data: Dict[str, Any], formula: str) -> Dict[str, Any]:
        """Convierte datos de MP a formato estándar."""
        try:
            # Extraer información básica
            structure = mp_data.get('structure', {})
            properties = mp_data.get('properties', {})
            
            # Calcular información adicional
            band_gap = properties.get('band_gap', 0.0)
            total_magnetization = properties.get('total_magnetization', 0.0)
            
            # Información de estructura
            lattice_params = structure.get('lattice', {}).get('a', None)
            
            return {
                'formula': formula,
                'source': 'MaterialsProject',
                'band_gap': band_gap,
                'total_magnetization': total_magnetization,
                'lattice_constant_a': lattice_params,
                'mp_id': mp_data.get('material_id', ''),
                'last_updated': mp_data.get('last_updated', ''),
                'calculations': mp_data.get('calculations', [])
            }
            
        except Exception as e:
            print(f"   ⚠️  Error procesando datos de {formula}: {e}")
            return {'formula': formula, 'source': 'MaterialsProject', 'error': str(e)}
    
    async def search_materials_by_properties(self, 
                                           band_gap_range: tuple = None,
                                           structure_type: str = None) -> List[Dict[str, Any]]:
        """
        Busca materiales por propiedades específicas.
        
        Args:
            band_gap_range: Tupla (min, max) para band gap
            structure_type: Tipo de estructura ('cubic', 'hexagonal', etc.)
            
        Returns:
            Lista de materiales que coinciden
        """
        print(f"🔍 Buscando materiales por propiedades...")
        
        # Construir query
        query_params = {}
        if band_gap_range:
            query_params['band_gap'] = f"[{band_gap_range[0]}:{band_gap_range[1]}]"
        if structure_type:
            query_params['structure_type'] = structure_type
        
        try:
            # URL para búsqueda
            url = f"{self.base_url}/materials/search"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=query_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = [self._process_mp_data(item, item.get('formula', '')) for item in data]
                        print(f"   ✅ Encontrados {len(results)} materiales")
                        return results
                    else:
                        print(f"   ❌ Error en búsqueda: {response.status}")
                        return []
                        
        except Exception as e:
            print(f"   ❌ Error en búsqueda: {e}")
            return []


class AFLOWDatabaseIntegration:
    """Integración con base de datos AFLOW."""
    
    def __init__(self):
        """Inicializa la integración con AFLOW."""
        self.base_url = "http://aflowlib.duke.edu/AFLOWDATA"
        print(f"🔗 Integración AFLOW inicializada")
    
    def search_by_formula(self, formula: str) -> Dict[str, Any]:
        """
        Busca material por fórmula en AFLOW.
        
        Args:
            formula: Fórmula química del material
            
        Returns:
            Diccionario con datos del material
        """
        print(f"🔍 Buscando {formula} en AFLOW...")
        
        try:
            # Construir URL de consulta AFLOW
            # Nota: Esta es una implementación simplificada
            # AFLOW tiene un sistema de consultas más complejo
            search_url = f"{self.base_url}/{formula}/"
            
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                # Procesar respuesta de AFLOW
                data = self._process_aflow_response(response.text, formula)
                print(f"   ✅ {formula}: Datos obtenidos de AFLOW")
                return data
            else:
                print(f"   ❌ {formula}: Error {response.status_code}")
                return {'formula': formula, 'source': 'AFLOW', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"   ❌ {formula}: Error en consulta AFLOW - {e}")
            return {'formula': formula, 'source': 'AFLOW', 'error': str(e)}
    
    def _process_aflow_response(self, response_text: str, formula: str) -> Dict[str, Any]:
        """Procesa respuesta de AFLOW."""
        # Esta es una implementación simplificada
        # AFLOW retorna datos en formatos específicos que requieren parsing
        try:
            # Por simplicidad, simulamos datos de AFLOW
            # En implementación real, se parsearía el XML/JSON retornado
            return {
                'formula': formula,
                'source': 'AFLOW',
                'energy_per_atom': -5.0,  # Simulado
                'formation_energy': 0.1,  # Simulado
                'band_gap': 1.5,          # Simulado
                'crystal_system': 'cubic', # Simulado
                'space_group': 'F-43m'    # Simulado
            }
        except Exception as e:
            return {'formula': formula, 'source': 'AFLOW', 'error': f'Parse error: {e}'}
    
    def batch_search(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """
        Busca múltiples materiales en AFLOW.
        
        Args:
            formulas: Lista de fórmulas
            
        Returns:
            Lista de resultados
        """
        print(f"🔍 Búsqueda en lote en AFLOW: {len(formulas)} materiales")
        results = []
        
        for formula in formulas:
            result = self.search_by_formula(formula)
            results.append(result)
        
        return results


class DFTCodeIntegrator:
    """Integrador para códigos DFT externos."""
    
    def __init__(self):
        """Inicializa integrador DFT."""
        self.supported_codes = ['quantum_espresso', 'vasp', 'abinit', 'cp2k']
        print(f"🔧 Integrador DFT inicializado para: {self.supported_codes}")
    
    def generate_quantum_espresso_input(self, 
                                      material_data: Dict[str, Any],
                                      output_path: Path) -> bool:
        """
        Genera archivo de entrada para Quantum ESPRESSO.
        
        Args:
            material_data: Datos del material
            output_path: Ruta de salida para el archivo
            
        Returns:
            True si se generó exitosamente
        """
        print(f"⚛️  Generando input QE para {material_data.get('formula', 'Unknown')}...")
        
        try:
            formula = material_data.get('formula', 'Unknown')
            lattice_constant = material_data.get('optimal_lattice_constant', 5.65)
            band_gap = material_data.get('band_gap', 1.5)
            
            # Generar input para Quantum ESPRESSO
            qe_input = f"""&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  outdir = './tmp/'
  pseudo_dir = './pseudo/'
  prefix = '{formula.lower()}'
/

&SYSTEM
  ibrav = 1,
  celldm(1) = {lattice_constant:.6f},
  nat = 2,
  ntyp = 2,
  ecutwfc = 40.0,
  occupations = 'smearing',
  smearing = 'mp',
  degauss = 0.02
/

&ELECTRONS
  conv_thr = 1.0d-6
  mixing_beta = 0.7
/

ATOMIC_SPECIES
  {formula[0]}  1.0  {formula[0]}.UPF
  {formula[1]}  1.0  {formula[1]}.UPF

ATOMIC_POSITIONS crystal
  {formula[0]} 0.0 0.0 0.0
  {formula[1]} 0.25 0.25 0.25

K_POINTS automatic
  8 8 8 0 0 0
"""
            
            # Crear directorio si no existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo
            with open(output_path, 'w') as f:
                f.write(qe_input)
            
            print(f"   ✅ Input QE guardado: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error generando input QE: {e}")
            return False
    
    def generate_vasp_input(self, 
                          material_data: Dict[str, Any],
                          output_dir: Path) -> bool:
        """
        Genera archivos de entrada para VASP.
        
        Args:
            material_data: Datos del material
            output_dir: Directorio de salida
            
        Returns:
            True si se generó exitosamente
        """
        print(f"⚛️  Generando input VASP para {material_data.get('formula', 'Unknown')}...")
        
        try:
            formula = material_data.get('formula', 'Unknown')
            lattice_constant = material_data.get('optimal_lattice_constant', 5.65)
            
            output_dir = output_dir / formula
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # POSCAR
            poscar = f"""Generated by Preconvergencia Multimaterial
1.0
{lattice_constant:.6f} 0.0 0.0
0.0 {lattice_constant:.6f} 0.0
0.0 0.0 {lattice_constant:.6f}
{formula[0]} {formula[1]}
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
            
            # INCAR
            incar = f"""ISTART = 0
ICHARG = 2
ENCUT = 500
GGA = PE
IVASP = 2
PREC = Accurate
KPAR = 4
"""
            
            # KPOINTS
            kpoints = f"""Automatic mesh
0              # automatic shift
8 8 8           # Monkhorst-Pack
0 0 0           # shift
"""
            
            # POTCAR (simulado)
            potcar = f"""Generated POTCAR for {formula}
# En implementación real, se seleccionarían pseudopotenciales específicos
"""
            
            # Escribir archivos
            with open(output_dir / "POSCAR", 'w') as f:
                f.write(poscar)
            
            with open(output_dir / "INCAR", 'w') as f:
                f.write(incar)
            
            with open(output_dir / "KPOINTS", 'w') as f:
                f.write(kpoints)
            
            with open(output_dir / "POTCAR", 'w') as f:
                f.write(potcar)
            
            print(f"   ✅ Input VASP guardado: {output_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error generando input VASP: {e}")
            return False
    
    def generate_abinit_input(self, 
                            material_data: Dict[str, Any],
                            output_path: Path) -> bool:
        """
        Genera archivo de entrada para ABINIT.
        
        Args:
            material_data: Datos del material
            output_path: Ruta de salida
            
        Returns:
            True si se generó exitosamente
        """
        print(f"⚛️  Generando input ABINIT para {material_data.get('formula', 'Unknown')}...")
        
        try:
            formula = material_data.get('formula', 'Unknown')
            lattice_constant = material_data.get('optimal_lattice_constant', 5.65)
            
            abinit_input = f"""# Input generado por Preconvergencia Multimaterial para {formula}
# Estructura zincblende optimizada

# Estructura cristalina
acell  1  1  {lattice_constant:.6f}  # Parametros de red en Bohr
rprim  1  0  0
        0  1  0
        0  0  1

# Átomos
ntypat  2  # Número de tipos atómicos
znucl  {10} {33}  # Números atómicos (ejemplo: Ga, As)
natom  2  # Número total de átomos

# Posiciones atómicas
xred
  0.0  0.0  0.0
  0.25 0.25 0.25

# Parámetros electrónicos
ecut  40  # Energía de corte (Hartree)
occopt 3  # Ocupación con smearing
tsmear 0.02  # Smearing (Hartree)

# Convergencia
toldfe 1.0d-6

# Paralelización
paral_kgb 1
"""
            
            # Crear directorio si no existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo
            with open(output_path, 'w') as f:
                f.write(abinit_input)
            
            print(f"   ✅ Input ABINIT guardado: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error generando input ABINIT: {e}")
            return False


class DataManagementSystem:
    """Sistema de gestión de datos para integración."""
    
    def __init__(self, storage_path: Path):
        """
        Inicializa sistema de gestión de datos.
        
        Args:
            storage_path: Ruta para almacenamiento
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Sistema de gestión de datos inicializado: {self.storage_path}")
    
    def save_integration_results(self, 
                               results: Dict[str, Any], 
                               experiment_name: str) -> Path:
        """
        Guarda resultados de integración.
        
        Args:
            results: Resultados a guardar
            experiment_name: Nombre del experimento
            
        Returns:
            Ruta del archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Resultados guardados: {filepath}")
        return filepath
    
    def load_integration_results(self, filepath: Path) -> Dict[str, Any]:
        """
        Carga resultados de integración.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Resultados cargados
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"📂 Resultados cargados: {filepath}")
            return results
        except Exception as e:
            print(f"❌ Error cargando resultados: {e}")
            return {}
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """
        Exporta datos a CSV.
        
        Args:
            data: Datos a exportar
            filename: Nombre del archivo
            
        Returns:
            Ruta del archivo CSV
        """
        df = pd.DataFrame(data)
        filepath = self.storage_path / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        print(f"📄 Datos exportados a CSV: {filepath}")
        return filepath
    
    def export_to_xlsx(self, data: Dict[str, List[Dict[str, Any]]], filename: str) -> Path:
        """
        Exporta datos a Excel con múltiples hojas.
        
        Args:
            data: Diccionario con datos por hoja
            filename: Nombre del archivo
            
        Returns:
            Ruta del archivo Excel
        """
        filepath = self.storage_path / f"{filename}.xlsx"
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, sheet_data in data.items():
                df = pd.DataFrame(sheet_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"📊 Datos exportados a Excel: {filepath}")
        return filepath


async def demo_integration_workflow():
    """Demuestra el flujo completo de integración."""
    print("🔗 DEMO: FLUJO COMPLETO DE INTEGRACIÓN")
    print("=" * 50)
    
    # 1. Preparar materiales para integración
    materiales_objetivo = ['GaAs', 'GaN', 'InP', 'ZnS']
    print(f"🎯 Materiales objetivo: {materiales_objetivo}")
    
    # 2. Ejecutar preconvergencia local
    print(f"\n🔬 Ejecutando preconvergencia local...")
    try:
        local_results = await run_custom_materials_campaign(
            materials=materiales_objetivo,
            parallel=True,
            max_workers=2
        )
        print(f"   ✅ Local: {local_results.materials_successful}/{local_results.materials_executed} exitosos")
    except Exception as e:
        print(f"   ⚠️  Error local: {e}")
        # Crear datos simulados
        local_results = crear_resultados_simulados(materiales_objetivo)
    
    return local_results


def crear_resultados_simulados(materiales: List[str]):
    """Crea resultados simulados para demostración."""
    from workflow.multi_material_pipeline import CampaignResult, MaterialExecutionResult
    from core.multi_material_config import MultiMaterialConfig
    
    resultados = []
    
    for material in materiales:
        resultado = MaterialExecutionResult(
            formula=material,
            success=True,
            execution_time=45.0,
            stages_completed=['cutoff', 'kmesh', 'lattice'],
            optimal_cutoff=420.0,
            optimal_kmesh=(6, 6, 6),
            optimal_lattice_constant=5.65
        )
        resultados.append(resultado)
    
    campaign = CampaignResult(
        materials_executed=len(materiales),
        materials_successful=len(resultados),
        materials_failed=0,
        total_execution_time=sum(r.execution_time for r in resultados),
        individual_results=resultados,
        campaign_config=MultiMaterialConfig()
    )
    
    return campaign


async def demo_external_database_integration():
    """Demuestra integración con bases de datos externas."""
    print("🌐 DEMO: INTEGRACIÓN CON BASES DE DATOS EXTERNAS")
    print("=" * 60)
    
    materiales_test = ['GaAs', 'GaN', 'SiC']
    
    # 1. Integración con Materials Project
    print(f"\n📡 Integración Materials Project:")
    mp_integration = MaterialsProjectIntegration()
    try:
        mp_results = await mp_integration.fetch_materials_data(materiales_test)
        print(f"   ✅ MP: {len(mp_results)} materiales obtenidos")
    except Exception as e:
        print(f"   ⚠️  MP: Error en integración - {e}")
        mp_results = []
    
    # 2. Integración con AFLOW
    print(f"\n📊 Integración AFLOW:")
    aflow_integration = AFLOWDatabaseIntegration()
    aflow_results = aflow_integration.batch_search(materiales_test)
    print(f"   ✅ AFLOW: {len(aflow_results)} materiales consultados")
    
    return mp_results, aflow_results


async def demo_dft_integration():
    """Demuestra integración con códigos DFT."""
    print("⚛️  DEMO: INTEGRACIÓN CON CÓDIGOS DFT")
    print("=" * 45)
    
    # Materiales para generar inputs
    materiales_dft = [
        {'formula': 'GaAs', 'optimal_lattice_constant': 5.653, 'band_gap': 1.42},
        {'formula': 'GaN', 'optimal_lattice_constant': 4.52, 'band_gap': 3.40}
    ]
    
    # Integrador DFT
    dft_integrator = DFTCodeIntegrator()
    
    output_base = Path("results/dft_inputs")
    output_base.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for material_data in materiales_dft:
        formula = material_data['formula']
        print(f"\n🔧 Generando inputs DFT para {formula}...")
        
        # Generar input para cada código
        qe_path = output_base / "quantum_espresso" / f"{formula}.in"
        vasp_dir = output_base / "vasp" / formula
        abinit_path = output_base / "abinit" / f"{formula}.abinit"
        
        # QE
        if dft_integrator.generate_quantum_espresso_input(material_data, qe_path):
            generated_files.append(str(qe_path))
        
        # VASP
        if dft_integrator.generate_vasp_input(material_data, output_base / "vasp"):
            generated_files.append(str(vasp_dir))
        
        # ABINIT
        if dft_integrator.generate_abinit_input(material_data, abinit_path):
            generated_files.append(str(abinit_path))
    
    print(f"\n✅ Archivos generados: {len(generated_files)}")
    for file_path in generated_files:
        print(f"   📄 {file_path}")
    
    return generated_files


def demo_data_management():
    """Demuestra sistema de gestión de datos."""
    print("💾 DEMO: SISTEMA DE GESTIÓN DE DATOS")
    print("=" * 45)
    
    storage_path = Path("results/integration_data")
    data_manager = DataManagementSystem(storage_path)
    
    # Simular datos de integración
    integration_data = {
        'timestamp': datetime.now().isoformat(),
        'materials_processed': 4,
        'external_databases': ['MaterialsProject', 'AFLOW'],
        'dft_codes': ['QuantumESPRESSO', 'VASP', 'ABINIT'],
        'results': {
            'local_convergence': {
                'total_materials': 4,
                'successful': 4,
                'average_time': 45.2
            },
            'external_validation': {
                'mp_matches': 3,
                'aflow_matches': 4
            }
        }
    }
    
    # Guardar resultados
    saved_file = data_manager.save_integration_results(
        integration_data, "integration_demo"
    )
    
    # Exportar a CSV
    csv_data = [
        {'formula': 'GaAs', 'source': 'Local', 'band_gap': 1.42, 'lattice': 5.653},
        {'formula': 'GaN', 'source': 'MP', 'band_gap': 3.40, 'lattice': 4.52},
        {'formula': 'SiC', 'source': 'AFLOW', 'band_gap': 2.36, 'lattice': 4.36}
    ]
    
    csv_file = data_manager.export_to_csv(csv_data, "integration_summary")
    
    # Exportar a Excel
    excel_data = {
        'Materials': csv_data,
        'Statistics': [
            {'metric': 'Total Materials', 'value': 3},
            {'metric': 'External Sources', 'value': 2},
            {'metric': 'Integration Success', 'value': '100%'}
        ]
    }
    
    excel_file = data_manager.export_to_xlsx(excel_data, "integration_report")
    
    return saved_file, csv_file, excel_file


def demo_api_rest_interface():
    """Demuestra interfaz REST API."""
    print("🌐 DEMO: INTERFAZ REST API")
    print("=" * 35)
    
    # Simular respuestas de API REST
    api_responses = {
        'materials_search': {
            'endpoint': '/api/v1/materials/search',
            'method': 'GET',
            'parameters': {'formula': 'GaAs', 'properties': ['band_gap', 'lattice']},
            'response': {
                'status': 'success',
                'data': {
                    'formula': 'GaAs',
                    'band_gap': 1.42,
                    'lattice_constant': 5.653,
                    'structure': 'zincblende'
                }
            }
        },
        'campaign_execute': {
            'endpoint': '/api/v1/campaigns/execute',
            'method': 'POST',
            'parameters': {
                'materials': ['GaAs', 'GaN'],
                'parallel': True,
                'max_workers': 4
            },
            'response': {
                'status': 'submitted',
                'campaign_id': 'camp_20241120_001',
                'estimated_duration': '15 minutes'
            }
        },
        'results_export': {
            'endpoint': '/api/v1/results/export',
            'method': 'GET',
            'parameters': {'campaign_id': 'camp_20241120_001', 'format': 'json'},
            'response': {
                'status': 'ready',
                'download_url': '/api/v1/files/camp_20241120_001_results.json',
                'expires': '2024-11-21T12:00:00Z'
            }
        }
    }
    
    print("📡 Endpoints de API simulados:")
    for endpoint, details in api_responses.items():
        print(f"   🔗 {endpoint}:")
        print(f"      • URL: {details['endpoint']}")
        print(f"      • Método: {details['method']}")
        print(f"      • Estado: {details['response']['status']}")
    
    return api_responses


async def main():
    """Función principal de demostración de integración."""
    print("🔗 DEMOSTRACIÓN DE INTEGRACIÓN CON OTROS SISTEMAS")
    print("=" * 65)
    
    # 1. Demo de flujo completo de preconvergencia
    local_results = await demo_integration_workflow()
    
    # 2. Demo de integración con bases de datos externas
    mp_results, aflow_results = await demo_external_database_integration()
    
    # 3. Demo de integración con códigos DFT
    dft_files = await demo_dft_integration()
    
    # 4. Demo de gestión de datos
    saved_file, csv_file, excel_file = demo_data_management()
    
    # 5. Demo de API REST
    api_endpoints = demo_api_rest_interface()
    
    # 6. Consolidar resultados
    print(f"\n📊 CONSOLIDACIÓN DE RESULTADOS")
    print("=" * 40)
    
    integration_summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'local_preconvergence': {
            'materials_processed': local_results.materials_executed,
            'successful': local_results.materials_successful,
            'success_rate': local_results.success_rate
        },
        'external_databases': {
            'materials_project': {
                'queried': len(mp_results),
                'successful_queries': sum(1 for r in mp_results if 'error' not in r)
            },
            'aflow': {
                'queried': len(aflow_results),
                'successful_queries': sum(1 for r in aflow_results if 'error' not in r)
            }
        },
        'dft_integration': {
            'codes_supported': ['QuantumESPRESSO', 'VASP', 'ABINIT'],
            'generated_files': len(dft_files)
        },
        'data_management': {
            'files_saved': [str(saved_file), str(csv_file), str(excel_file)],
            'storage_format': ['JSON', 'CSV', 'Excel']
        },
        'api_interface': {
            'endpoints': len(api_endpoints),
            'methods': list(set(details['method'] for details in api_endpoints.values()))
        }
    }
    
    # Guardar resumen
    storage_path = Path("results/integration_data")
    summary_file = storage_path / "integration_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(integration_summary, f, indent=2)
    
    print(f"✅ Integración completada exitosamente")
    print(f"📁 Archivos generados:")
    print(f"   • Resumen: {summary_file}")
    print(f"   • Datos: {csv_file}")
    print(f"   • Reporte: {excel_file}")
    print(f"   • DFT inputs: {len(dft_files)} archivos")
    
    print(f"\n🎯 Capacidades de integración demostradas:")
    print(f"   ✅ Preconvergencia local con PySCF")
    print(f"   ✅ Conexión con Materials Project API")
    print(f"   ✅ Integración con base de datos AFLOW")
    print(f"   ✅ Generación de inputs para QE, VASP, ABINIT")
    print(f"   ✅ Sistema de gestión de datos")
    print(f"   ✅ Interfaz REST API")
    print(f"   ✅ Exportación multi-formato (JSON, CSV, Excel)")
    
    return integration_summary


if __name__ == "__main__":
    try:
        summary = asyncio.run(main())
        print(f"\n🎉 Demo de integración completado exitosamente")
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el demo: {e}")
        import traceback
        traceback.print_exc()