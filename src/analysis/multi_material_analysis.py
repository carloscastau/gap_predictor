# src/analysis/multi_material_analysis.py
"""Análisis avanzado de resultados de múltiples materiales semiconductores."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from ..workflow.multi_material_pipeline import CampaignResult, MaterialExecutionResult
from ..models.semiconductor_database import (
    BinarySemiconductor, 
    SemiconductorType,
    SEMICONDUCTOR_DB
)
from ..utils.logging import StructuredLogger


logger = logging.getLogger(__name__)

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ParameterComparison:
    """Comparación de parámetros entre materiales."""
    parameter_name: str
    values: List[float]
    materials: List[str]
    statistics: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    outliers: List[Tuple[str, float]] = None
    
    def get_summary(self) -> dict:
        """Obtiene resumen estadístico."""
        return {
            'parameter': self.parameter_name,
            'count': len(self.values),
            'mean': np.mean(self.values),
            'std': np.std(self.values),
            'min': np.min(self.values),
            'max': np.max(self.values),
            'median': np.median(self.values),
            'q25': np.percentile(self.values, 25),
            'q75': np.percentile(self.values, 75),
            'outliers': self.outliers or []
        }


@dataclass
class MaterialGroupAnalysis:
    """Análisis agrupado por tipo de semiconductor."""
    semiconductor_type: SemiconductorType
    materials: List[str]
    count: int
    parameter_comparisons: List[ParameterComparison]
    success_rate: float
    average_execution_time: float
    group_statistics: Dict[str, Any]
    
    def get_summary(self) -> dict:
        """Obtiene resumen del grupo."""
        return {
            'type': self.semiconductor_type.value,
            'count': self.count,
            'success_rate': self.success_rate,
            'average_execution_time': self.average_execution_time,
            'materials': self.materials,
            'group_statistics': self.group_statistics
        }


@dataclass
class MultiMaterialAnalysisReport:
    """Reporte completo de análisis multimaterial."""
    campaign_result: CampaignResult
    material_analyses: Dict[str, Dict[str, Any]]
    parameter_comparisons: List[ParameterComparison]
    group_analyses: List[MaterialGroupAnalysis]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]
    visualizations_created: List[str]
    
    def get_executive_summary(self) -> dict:
        """Obtiene resumen ejecutivo."""
        successful_materials = self.campaign_result.get_successful_materials()
        failed_materials = self.campaign_result.get_failed_materials()
        
        return {
            'campaign_overview': {
                'total_materials': self.campaign_result.materials_executed,
                'successful': len(successful_materials),
                'failed': len(failed_materials),
                'success_rate': self.campaign_result.success_rate,
                'total_time': self.campaign_result.total_execution_time,
                'average_time_per_material': self.campaign_result.average_execution_time
            },
            'key_findings': {
                'successful_materials': successful_materials,
                'failed_materials': failed_materials,
                'fastest_material': self._get_fastest_material(),
                'slowest_material': self._get_slowest_material(),
                'most_convergent_cutoff': self._get_most_convergent_cutoff(),
                'optimal_cutoff_range': self._get_cutoff_range(),
                'optimal_lattice_range': self._get_lattice_range()
            },
            'recommendations': self.recommendations
        }
    
    def _get_fastest_material(self) -> Optional[str]:
        """Obtiene material con menor tiempo de ejecución."""
        execution_times = [
            (result.formula, result.execution_time)
            for result in self.campaign_result.individual_results
            if result.success
        ]
        if not execution_times:
            return None
        return min(execution_times, key=lambda x: x[1])[0]
    
    def _get_slowest_material(self) -> Optional[str]:
        """Obtiene material con mayor tiempo de ejecución."""
        execution_times = [
            (result.formula, result.execution_time)
            for result in self.campaign_result.individual_results
            if result.success
        ]
        if not execution_times:
            return None
        return max(execution_times, key=lambda x: x[1])[0]
    
    def _get_most_convergent_cutoff(self) -> Optional[float]:
        """Obtiene cutoff más convergente (menor valor)."""
        cutoffs = [
            result.optimal_cutoff
            for result in self.campaign_result.individual_results
            if result.success and result.optimal_cutoff
        ]
        return min(cutoffs) if cutoffs else None
    
    def _get_cutoff_range(self) -> Optional[Tuple[float, float]]:
        """Obtiene rango de cutoffs óptimos."""
        cutoffs = [
            result.optimal_cutoff
            for result in self.campaign_result.individual_results
            if result.success and result.optimal_cutoff
        ]
        return (min(cutoffs), max(cutoffs)) if cutoffs else None
    
    def _get_lattice_range(self) -> Optional[Tuple[float, float]]:
        """Obtiene rango de constantes de red óptimas."""
        lattices = [
            result.optimal_lattice_constant
            for result in self.campaign_result.individual_results
            if result.success and result.optimal_lattice_constant
        ]
        return (min(lattices), max(lattices)) if lattices else None


class MultiMaterialAnalyzer:
    """Analizador principal para resultados multimaterial."""
    
    def __init__(self, enable_visualizations: bool = True):
        """
        Inicializa el analizador.
        
        Args:
            enable_visualizations: Habilitar creación de visualizaciones
        """
        self.enable_visualizations = enable_visualizations
        self.logger = StructuredLogger("MultiMaterialAnalyzer")
        
        # Configurar estilo de gráficos
        if self.enable_visualizations:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def analyze_campaign_results(self, 
                               campaign_result: CampaignResult,
                               output_dir: Optional[Path] = None) -> MultiMaterialAnalysisReport:
        """
        Analiza resultados completos de una campaña.
        
        Args:
            campaign_result: Resultado de la campaña
            output_dir: Directorio de salida (crea uno si None)
            
        Returns:
            Reporte completo de análisis
        """
        if output_dir is None:
            output_dir = Path("analysis_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Analizando campaña con {campaign_result.materials_executed} materiales")
        
        # Análisis individual de materiales
        material_analyses = self._analyze_individual_materials(campaign_result)
        
        # Comparación de parámetros
        parameter_comparisons = self._compare_parameters(campaign_result)
        
        # Análisis por grupos
        group_analyses = self._analyze_by_groups(campaign_result)
        
        # Tests estadísticos
        statistical_tests = self._perform_statistical_tests(parameter_comparisons)
        
        # Recomendaciones
        recommendations = self._generate_recommendations(
            campaign_result, material_analyses, parameter_comparisons, group_analyses
        )
        
        # Visualizaciones
        visualizations_created = []
        if self.enable_visualizations:
            visualizations_created = self._create_visualizations(
                campaign_result, parameter_comparisons, group_analyses, output_dir
            )
        
        # Crear reporte
        report = MultiMaterialAnalysisReport(
            campaign_result=campaign_result,
            material_analyses=material_analyses,
            parameter_comparisons=parameter_comparisons,
            group_analyses=group_analyses,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            visualizations_created=visualizations_created
        )
        
        # Guardar reporte
        self._save_analysis_report(report, output_dir)
        
        self.logger.info(f"Análisis completado. Reporte guardado en {output_dir}")
        
        return report
    
    def _analyze_individual_materials(self, campaign_result: CampaignResult) -> Dict[str, Dict[str, Any]]:
        """Analiza materiales individuales."""
        analyses = {}
        
        for result in campaign_result.individual_results:
            analysis = {
                'formula': result.formula,
                'success': result.success,
                'execution_time': result.execution_time,
                'stages_completed': result.stages_completed,
                'optimal_parameters': {}
            }
            
            if result.success:
                # Agregar parámetros óptimos
                if result.optimal_cutoff:
                    analysis['optimal_parameters']['cutoff'] = result.optimal_cutoff
                if result.optimal_kmesh:
                    analysis['optimal_parameters']['kmesh'] = result.optimal_kmesh
                if result.optimal_lattice_constant:
                    analysis['optimal_parameters']['lattice_constant'] = result.optimal_lattice_constant
                
                # Agregar información del semiconductor
                if result.formula in SEMICONDUCTOR_DB.semiconductors:
                    semiconductor = SEMICONDUCTOR_DB.semiconductors[result.formula]
                    analysis['semiconductor_info'] = {
                        'type': semiconductor.semiconductor_type.value,
                        'cation': semiconductor.cation.symbol,
                        'anion': semiconductor.anion.symbol,
                        'ionic_radius_ratio': semiconductor.ionic_radius_ratio,
                        'electronegativity_difference': semiconductor.electronegativity_difference
                    }
            
            analyses[result.formula] = analysis
        
        return analyses
    
    def _compare_parameters(self, campaign_result: CampaignResult) -> List[ParameterComparison]:
        """Compara parámetros óptimos entre materiales."""
        successful_results = [r for r in campaign_result.individual_results if r.success]
        
        comparisons = []
        
        # Comparar cutoffs
        cutoffs = [
            (r.formula, r.optimal_cutoff)
            for r in successful_results
            if r.optimal_cutoff
        ]
        
        if cutoffs:
            materials, values = zip(*cutoffs)
            comparison = ParameterComparison(
                parameter_name="optimal_cutoff",
                values=list(values),
                materials=list(materials),
                statistics=self._calculate_statistics(values),
                outliers=self._detect_outliers(values, materials)
            )
            comparisons.append(comparison)
        
        # Comparar constantes de red
        lattices = [
            (r.formula, r.optimal_lattice_constant)
            for r in successful_results
            if r.optimal_lattice_constant
        ]
        
        if lattices:
            materials, values = zip(*lattices)
            comparison = ParameterComparison(
                parameter_name="optimal_lattice_constant",
                values=list(values),
                materials=list(materials),
                statistics=self._calculate_statistics(values),
                outliers=self._detect_outliers(values, materials)
            )
            comparisons.append(comparison)
        
        # Comparar tiempos de ejecución
        execution_times = [
            (r.formula, r.execution_time)
            for r in successful_results
        ]
        
        if execution_times:
            materials, values = zip(*execution_times)
            comparison = ParameterComparison(
                parameter_name="execution_time",
                values=list(values),
                materials=list(materials),
                statistics=self._calculate_statistics(values),
                outliers=self._detect_outliers(values, materials)
            )
            comparisons.append(comparison)
        
        return comparisons
    
    def _analyze_by_groups(self, campaign_result: CampaignResult) -> List[MaterialGroupAnalysis]:
        """Analiza materiales agrupados por tipo."""
        groups = {}
        
        # Agrupar por tipo de semiconductor
        for result in campaign_result.individual_results:
            if not result.success:
                continue
            
            if result.formula not in SEMICONDUCTOR_DB.semiconductors:
                continue
            
            semiconductor = SEMICONDUCTOR_DB.semiconductors[result.formula]
            sem_type = semiconductor.semiconductor_type
            
            if sem_type not in groups:
                groups[sem_type] = []
            
            groups[sem_type].append(result)
        
        group_analyses = []
        
        for sem_type, results in groups.items():
            materials = [r.formula for r in results]
            success_count = len(results)
            total_count = len([
                r for r in campaign_result.individual_results
                if r.formula in materials
            ])
            
            # Calcular estadísticas del grupo
            execution_times = [r.execution_time for r in results]
            cutoffs = [r.optimal_cutoff for r in results if r.optimal_cutoff]
            lattices = [r.optimal_lattice_constant for r in results if r.optimal_lattice_constant]
            
            group_stats = {
                'execution_times': self._calculate_statistics(execution_times),
                'cutoffs': self._calculate_statistics(cutoffs) if cutoffs else {},
                'lattices': self._calculate_statistics(lattices) if lattices else {}
            }
            
            # Parámetros de comparación para el grupo
            parameter_comparisons = []
            if cutoffs:
                parameter_comparisons.append(ParameterComparison(
                    parameter_name="cutoff",
                    values=cutoffs,
                    materials=materials,
                    statistics=self._calculate_statistics(cutoffs)
                ))
            
            analysis = MaterialGroupAnalysis(
                semiconductor_type=sem_type,
                materials=materials,
                count=success_count,
                parameter_comparisons=parameter_comparisons,
                success_rate=(success_count / total_count * 100) if total_count > 0 else 0,
                average_execution_time=np.mean(execution_times) if execution_times else 0,
                group_statistics=group_stats
            )
            
            group_analyses.append(analysis)
        
        return group_analyses
    
    def _perform_statistical_tests(self, comparisons: List[ParameterComparison]) -> Dict[str, Any]:
        """Realiza tests estadísticos en los parámetros."""
        tests = {}
        
        for comparison in comparisons:
            if len(comparison.values) < 3:
                continue
            
            param_name = comparison.parameter_name
            values = comparison.values
            
            # Test de normalidad
            _, p_normal = stats.normaltest(values)
            
            # Test de correlación entre parámetros (si hay más de uno)
            if len(comparisons) > 1:
                correlations = {}
                for other in comparisons:
                    if other.parameter_name != param_name and len(other.values) == len(values):
                        corr_coef, p_corr = stats.pearsonr(values, other.values)
                        correlations[other.parameter_name] = {
                            'correlation': corr_coef,
                            'p_value': p_corr,
                            'significant': p_corr < 0.05
                        }
            else:
                correlations = {}
            
            tests[param_name] = {
                'normality_test': {
                    'statistic': _,
                    'p_value': p_normal,
                    'is_normal': p_normal > 0.05
                },
                'descriptive_stats': comparison.statistics,
                'correlations': correlations
            }
        
        return tests
    
    def _generate_recommendations(self, 
                                campaign_result: CampaignResult,
                                material_analyses: Dict[str, Dict[str, Any]],
                                parameter_comparisons: List[ParameterComparison],
                                group_analyses: List[MaterialGroupAnalysis]) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        successful_count = campaign_result.materials_successful
        total_count = campaign_result.materials_executed
        
        # Recomendaciones generales de la campaña
        if successful_count / total_count < 0.8:
            recommendations.append(
                f"Tasa de éxito baja ({successful_count}/{total_count}). "
                f"Revisar configuraciones de materiales fallidos."
            )
        
        if campaign_result.average_execution_time > 300:  # 5 minutos
            recommendations.append(
                f"Tiempo promedio de ejecución alto ({campaign_result.average_execution_time:.1f}s). "
                f"Considerar reducir parámetros de búsqueda o usar paralelización."
            )
        
        # Recomendaciones basadas en parámetros
        cutoff_comparison = next((c for c in parameter_comparisons if c.parameter_name == "optimal_cutoff"), None)
        if cutoff_comparison:
            cutoff_range = cutoff_comparison.statistics['max'] - cutoff_comparison.statistics['min']
            if cutoff_range > 100:
                recommendations.append(
                    f"Alto rango en cutoffs óptimos ({cutoff_range:.0f}). "
                    f"Considerar análisis por tipo de semiconductor."
                )
        
        # Recomendaciones por grupo
        for group_analysis in group_analyses:
            if group_analysis.success_rate < 70:
                recommendations.append(
                    f"Baja tasa de éxito en {group_analysis.semiconductor_type.value}: "
                    f"{group_analysis.success_rate:.1f}%. "
                    f"Revisar parámetros específicos del grupo."
                )
        
        # Recomendaciones de optimización
        if len(parameter_comparisons) > 1:
            recommendations.append(
                "Ejecutar análisis de correlación para identificar parámetros interdependientes."
            )
        
        if not recommendations:
            recommendations.append("Todos los materiales ejecutados exitosamente. Sistema funcionando óptimamente.")
        
        return recommendations
    
    def _create_visualizations(self, 
                             campaign_result: CampaignResult,
                             parameter_comparisons: List[ParameterComparison],
                             group_analyses: List[MaterialGroupAnalysis],
                             output_dir: Path) -> List[str]:
        """Crea visualizaciones de los resultados."""
        visualizations_created = []
        
        try:
            # 1. Gráfico de éxito por material
            self._plot_material_success_rate(campaign_result, output_dir)
            visualizations_created.append("material_success_rate.png")
            
            # 2. Comparación de parámetros
            if parameter_comparisons:
                self._plot_parameter_comparisons(parameter_comparisons, output_dir)
                visualizations_created.append("parameter_comparisons.png")
            
            # 3. Análisis por grupos
            if group_analyses:
                self._plot_group_analysis(group_analyses, output_dir)
                visualizations_created.append("group_analysis.png")
            
            # 4. Heatmap de correlaciones
            if len(parameter_comparisons) > 1:
                self._plot_correlation_heatmap(parameter_comparisons, output_dir)
                visualizations_created.append("correlation_heatmap.png")
            
            # 5. Distribución de tiempos de ejecución
            self._plot_execution_time_distribution(campaign_result, output_dir)
            visualizations_created.append("execution_time_distribution.png")
            
        except Exception as e:
            self.logger.warning(f"Error creando visualizaciones: {e}")
        
        return visualizations_created
    
    def _plot_material_success_rate(self, campaign_result: CampaignResult, output_dir: Path):
        """Plot tasa de éxito por material."""
        successful = campaign_result.get_successful_materials()
        failed = campaign_result.get_failed_materials()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de barras de éxito/fallo
        materials = successful + failed
        statuses = ['Exitoso'] * len(successful) + ['Fallido'] * len(failed)
        colors = ['green'] * len(successful) + ['red'] * len(failed)
        
        y_pos = np.arange(len(materials))
        ax1.barh(y_pos, [1] * len(materials), color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(materials)
        ax1.set_xlabel('Estado')
        ax1.set_title('Estado de Ejecución por Material')
        
        # Gráfico de torta de éxito
        success_count = len(successful)
        failure_count = len(failed)
        ax2.pie([success_count, failure_count], 
               labels=['Exitosos', 'Fallidos'],
               colors=['green', 'red'],
               autopct='%1.1f%%')
        ax2.set_title('Distribución de Éxito')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'material_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_comparisons(self, comparisons: List[ParameterComparison], output_dir: Path):
        """Plot comparaciones de parámetros."""
        n_params = len(comparisons)
        fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for i, comparison in enumerate(comparisons):
            ax = axes[i]
            
            # Box plot
            ax.boxplot(comparison.values, labels=[comparison.parameter_name])
            ax.set_ylabel(comparison.parameter_name)
            ax.set_title(f'Distribución de {comparison.parameter_name}')
            
            # Agregar puntos individuales
            x_pos = np.ones(len(comparison.values))
            ax.scatter(x_pos, comparison.values, alpha=0.6, s=50)
            
            # Marcar outliers
            if comparison.outliers:
                for material, value in comparison.outliers:
                    ax.annotate(material, (1, value), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_group_analysis(self, group_analyses: List[MaterialGroupAnalysis], output_dir: Path):
        """Plot análisis por grupos."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        group_names = [ga.semiconductor_type.value for ga in group_analyses]
        success_rates = [ga.success_rate for ga in group_analyses]
        avg_times = [ga.average_execution_time for ga in group_analyses]
        counts = [ga.count for ga in group_analyses]
        
        # Tasas de éxito
        ax1.bar(group_names, success_rates, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Tasa de Éxito (%)')
        ax1.set_title('Tasa de Éxito por Tipo de Semiconductor')
        ax1.set_ylim(0, 100)
        
        # Tiempos promedio
        ax2.bar(group_names, avg_times, color=['gold', 'orange'])
        ax2.set_ylabel('Tiempo Promedio (s)')
        ax2.set_title('Tiempo Promedio de Ejecución')
        
        # Número de materiales
        ax3.bar(group_names, counts, color=['lightgreen', 'mediumseagreen'])
        ax3.set_ylabel('Número de Materiales')
        ax3.set_title('Materiales Procesados Exitosamente')
        
        # Scatter plot tiempo vs éxito
        ax4.scatter(avg_times, success_rates, s=counts, alpha=0.7)
        for i, name in enumerate(group_names):
            ax4.annotate(name, (avg_times[i], success_rates[i]))
        ax4.set_xlabel('Tiempo Promedio (s)')
        ax4.set_ylabel('Tasa de Éxito (%)')
        ax4.set_title('Tiempo vs Tasa de Éxito')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'group_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, comparisons: List[ParameterComparison], output_dir: Path):
        """Plot heatmap de correlaciones."""
        # Crear matriz de correlación
        param_names = [c.parameter_name for c in comparisons]
        correlation_matrix = np.eye(len(param_names))  # Diagonal
        
        for i, comp1 in enumerate(comparisons):
            for j, comp2 in enumerate(comparisons):
                if i != j and len(comp1.values) == len(comp2.values):
                    corr, _ = stats.pearsonr(comp1.values, comp2.values)
                    correlation_matrix[i, j] = corr
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, 
                   xticklabels=param_names,
                   yticklabels=param_names,
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   ax=ax)
        ax.set_title('Matriz de Correlación entre Parámetros')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time_distribution(self, campaign_result: CampaignResult, output_dir: Path):
        """Plot distribución de tiempos de ejecución."""
        successful_results = [r for r in campaign_result.individual_results if r.success]
        
        if not successful_results:
            return
        
        execution_times = [r.execution_time for r in successful_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma
        ax1.hist(execution_times, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Tiempo de Ejecución (s)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Tiempos de Ejecución')
        ax1.axvline(np.mean(execution_times), color='red', linestyle='--', 
                   label=f'Media: {np.mean(execution_times):.1f}s')
        ax1.legend()
        
        # Box plot por tipo
        materials = [r.formula for r in successful_results]
        times = execution_times
        
        ax2.boxplot(times, labels=['Todos los Materiales'])
        ax2.set_ylabel('Tiempo de Ejecución (s)')
        ax2.set_title('Distribución de Tiempos')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calcula estadísticas descriptivas."""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
    
    def _detect_outliers(self, values: List[float], materials: List[str]) -> List[Tuple[str, float]]:
        """Detecta outliers usando IQR."""
        if len(values) < 4:
            return []
        
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = []
        for material, value in zip(materials, values):
            if value < lower_bound or value > upper_bound:
                outliers.append((material, value))
        
        return outliers
    
    def _save_analysis_report(self, report: MultiMaterialAnalysisReport, output_dir: Path):
        """Guarda reporte completo de análisis."""
        # Resumen ejecutivo
        executive_summary = report.get_executive_summary()
        
        # Datos completos
        analysis_data = {
            'timestamp': str(pd.Timestamp.now()),
            'executive_summary': executive_summary,
            'material_analyses': report.material_analyses,
            'parameter_comparisons': [
                {
                    'parameter': pc.parameter_name,
                    'materials': pc.materials,
                    'values': pc.values,
                    'statistics': pc.statistics,
                    'outliers': pc.outliers
                }
                for pc in report.parameter_comparisons
            ],
            'group_analyses': [ga.get_summary() for ga in report.group_analyses],
            'statistical_tests': report.statistical_tests,
            'recommendations': report.recommendations,
            'visualizations_created': report.visualizations_created
        }
        
        # Guardar JSON
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        # Guardar CSV de parámetros
        if report.parameter_comparisons:
            self._save_parameters_to_csv(report, output_dir)
        
        self.logger.info(f"Reporte de análisis guardado en {output_dir}")
    
    def _save_parameters_to_csv(self, report: MultiMaterialAnalysisReport, output_dir: Path):
        """Guarda parámetros en formato CSV."""
        data = []
        
        for result in report.campaign_result.individual_results:
            row = {
                'material': result.formula,
                'success': result.success,
                'execution_time': result.execution_time,
                'optimal_cutoff': result.optimal_cutoff,
                'optimal_kmesh': str(result.optimal_kmesh) if result.optimal_kmesh else None,
                'optimal_lattice_constant': result.optimal_lattice_constant
            }
            
            # Agregar información del semiconductor si está disponible
            if result.formula in SEMICONDUCTOR_DB.semiconductors:
                sem = SEMICONDUCTOR_DB.semiconductors[result.formula]
                row.update({
                    'semiconductor_type': sem.semiconductor_type.value,
                    'cation': sem.cation.symbol,
                    'anion': sem.anion.symbol,
                    'ionic_radius_ratio': sem.ionic_radius_ratio,
                    'electronegativity_difference': sem.electronegativity_difference
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_dir / 'materials_analysis.csv', index=False)


# Funciones de conveniencia para análisis rápido

def analyze_campaign_quick(campaign_result: CampaignResult, output_dir: Path = None) -> Dict[str, Any]:
    """
    Análisis rápido de resultados de campaña.
    
    Args:
        campaign_result: Resultado de la campaña
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con resumen de análisis
    """
    analyzer = MultiMaterialAnalyzer(enable_visualizations=False)
    report = analyzer.analyze_campaign_results(campaign_result, output_dir)
    return report.get_executive_summary()


def compare_two_campaigns(campaign1: CampaignResult, 
                        campaign2: CampaignResult,
                        output_dir: Path = None) -> Dict[str, Any]:
    """
    Compara resultados de dos campañas diferentes.
    
    Args:
        campaign1: Primera campaña
        campaign2: Segunda campaña
        output_dir: Directorio de salida
        
    Returns:
        Comparación detallada
    """
    if output_dir is None:
        output_dir = Path("campaign_comparison")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analizar ambas campañas
    analyzer = MultiMaterialAnalyzer()
    report1 = analyzer.analyze_campaign_results(campaign1, output_dir / "campaign1")
    report2 = analyzer.analyze_campaign_results(campaign2, output_dir / "campaign2")
    
    # Crear comparación
    comparison = {
        'campaign1_summary': report1.get_executive_summary(),
        'campaign2_summary': report2.get_executive_summary(),
        'performance_comparison': {
            'success_rate_diff': report1.campaign_result.success_rate - report2.campaign_result.success_rate,
            'time_diff': report1.campaign_result.average_execution_time - report2.campaign_result.average_execution_time,
            'best_performer': 'campaign1' if report1.campaign_result.success_rate > report2.campaign_result.success_rate else 'campaign2'
        }
    }
    
    # Guardar comparación
    with open(output_dir / 'campaign_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    return comparison