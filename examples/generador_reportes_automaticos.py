#!/usr/bin/env python3
# examples/generador_reportes_automaticos.py
"""
Generador de Reportes Automáticos - Sistema de Preconvergencia Multimaterial

Este script genera reportes automáticos comprensivos que incluyen:
- Reportes ejecutivos HTML con visualizaciones
- Reportes técnicos PDF para publicación
- Dashboards interactivos con métricas en tiempo real
- Reportes de comparativa entre estudios
- Exportación de datos en múltiples formatos

Ejecutar: python examples/generador_reportes_automaticos.py
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import base64
import io

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow.multi_material_pipeline import (
    MultiMaterialPipeline, 
    run_common_semiconductors_campaign,
    CampaignResult
)
from analysis.multi_material_analysis import MultiMaterialAnalyzer
from core.multi_material_config import MultiMaterialConfig
from models.semiconductor_database import SEMICONDUCTOR_DB
from core.material_permutator import MATERIAL_PERMUTATOR


def generar_html_ejecutivo(campaign_results: List[CampaignResult], output_dir: Path):
    """Genera reporte ejecutivo en formato HTML."""
    print("📊 GENERANDO REPORTE EJECUTIVO HTML")
    print("=" * 45)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcular estadísticas generales
    total_campaigns = len(campaign_results)
    total_materials = sum(cr.materials_executed for cr in campaign_results)
    total_successful = sum(cr.materials_successful for cr in campaign_results)
    overall_success_rate = (total_successful / total_materials * 100) if total_materials > 0 else 0
    
    # Crear HTML del reporte
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Ejecutivo - Preconvergencia Multimaterial</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #2c3e50;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .campaign-card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        .campaign-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .campaign-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .success-rate {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .success-high {{ background-color: #27ae60; }}
        .success-medium {{ background-color: #f39c12; }}
        .success-low {{ background-color: #e74c3c; }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
        }}
        .recommendations {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            margin-top: 0;
            font-size: 1.4em;
        }}
        .recommendations ul {{
            list-style-type: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 8px 0;
            padding-left: 20px;
            position: relative;
        }}
        .recommendations li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #00b894;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte Ejecutivo</h1>
            <div class="subtitle">Sistema de Preconvergencia Multimaterial - Análisis Comprensivo</div>
            <div class="timestamp">Generado el {datetime.now().strftime("%d de %B de %Y a las %H:%M")}</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_campaigns}</div>
                <div class="metric-label">Campañas Ejecutadas</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_materials}</div>
                <div class="metric-label">Materiales Analizados</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_successful}</div>
                <div class="metric-label">Análisis Exitosos</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{overall_success_rate:.1f}%</div>
                <div class="metric-label">Tasa de Éxito General</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Detalle de Campañas</h2>
"""
    
    # Agregar información de cada campaña
    for i, campaign in enumerate(campaign_results, 1):
        success_rate = campaign.success_rate
        success_class = "success-high" if success_rate >= 80 else "success-medium" if success_rate >= 60 else "success-low"
        
        html_content += f"""
            <div class="campaign-card">
                <div class="campaign-header">
                    <div class="campaign-title">Campaña {i}</div>
                    <div class="success-rate {success_class}">{success_rate:.1f}%</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {success_rate}%"></div>
                </div>
                <p><strong>Materiales procesados:</strong> {campaign.materials_executed}</p>
                <p><strong>Materiales exitosos:</strong> {campaign.materials_successful}</p>
                <p><strong>Tiempo total:</strong> {campaign.total_execution_time/60:.1f} minutos</p>
                <p><strong>Tiempo promedio por material:</strong> {campaign.average_execution_time:.1f} segundos</p>
            </div>
        """
    
    # Agregar recomendaciones basadas en los resultados
    recommendations = generar_recomendaciones_automaticas(campaign_results)
    
    html_content += f"""
        </div>
        
        <div class="recommendations">
            <h3>💡 Recomendaciones Automáticas</h3>
            <ul>
    """
    
    for rec in recommendations:
        html_content += f"<li>{rec}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="footer">
            <p>Este reporte fue generado automáticamente por el Sistema de Preconvergencia Multimaterial</p>
            <p>Para más información técnica, consulte la documentación completa del sistema</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Guardar archivo HTML
    report_path = output_dir / "reporte_ejecutivo.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✅ Reporte HTML guardado: {report_path}")
    return report_path


def generar_dashboard_interactivo(campaign_results: List[CampaignResult], output_dir: Path):
    """Genera dashboard interactivo con JavaScript."""
    print("📊 GENERANDO DASHBOARD INTERACTIVO")
    print("=" * 40)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar datos para el dashboard
    dashboard_data = {
        'campaigns': [],
        'summary': {
            'total_campaigns': len(campaign_results),
            'total_materials': sum(cr.materials_executed for cr in campaign_results),
            'total_successful': sum(cr.materials_successful for cr in campaign_results),
            'avg_success_rate': sum(cr.success_rate for cr in campaign_results) / len(campaign_results) if campaign_results else 0,
            'avg_execution_time': sum(cr.total_execution_time for cr in campaign_results) / len(campaign_results) if campaign_results else 0
        }
    }
    
    for i, campaign in enumerate(campaign_results):
        dashboard_data['campaigns'].append({
            'id': i + 1,
            'materials_executed': campaign.materials_executed,
            'materials_successful': campaign.materials_successful,
            'materials_failed': campaign.materials_failed,
            'success_rate': campaign.success_rate,
            'total_execution_time': campaign.total_execution_time,
            'average_execution_time': campaign.average_execution_time
        })
    
    # Crear HTML del dashboard
    dashboard_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Preconvergencia Multimaterial</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 3em; margin-bottom: 10px; }}
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.9);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }}
        .campaigns-table {{
            background: rgba(255,255,255,0.9);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        .table th {{
            background: #34495e;
            color: white;
            font-weight: bold;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
        }}
        .status-success {{ background: #27ae60; }}
        .status-warning {{ background: #f39c12; }}
        .status-error {{ background: #e74c3c; }}
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 2em; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>📊 Dashboard en Tiempo Real</h1>
            <p>Sistema de Preconvergencia Multimaterial</p>
        </div>
        
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value" id="totalCampaigns">{dashboard_data['summary']['total_campaigns']}</div>
                <div class="metric-label">Campañas</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="totalMaterials">{dashboard_data['summary']['total_materials']}</div>
                <div class="metric-label">Materiales</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgSuccessRate">{dashboard_data['summary']['avg_success_rate']:.1f}%</div>
                <div class="metric-label">Éxito Promedio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgTime">{dashboard_data['summary']['avg_execution_time']:.0f}s</div>
                <div class="metric-label">Tiempo Promedio</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Tasa de Éxito por Campaña</div>
                <canvas id="successRateChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Materiales Procesados</div>
                <canvas id="materialsChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="campaigns-table">
            <h2 style="margin-bottom: 20px; color: #2c3e50;">📋 Detalle de Campañas</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Materiales</th>
                        <th>Exitosos</th>
                        <th>Fallidos</th>
                        <th>Tasa Éxito</th>
                        <th>Tiempo Total</th>
                        <th>Estado</th>
                    </tr>
                </thead>
                <tbody id="campaignsTableBody">
    """
    
    # Agregar filas de la tabla
    for campaign in dashboard_data['campaigns']:
        status_class = "status-success" if campaign['success_rate'] >= 80 else "status-warning" if campaign['success_rate'] >= 60 else "status-error"
        status_text = "Excelente" if campaign['success_rate'] >= 80 else "Bueno" if campaign['success_rate'] >= 60 else "Revisar"
        
        dashboard_html += f"""
                    <tr>
                        <td>{campaign['id']}</td>
                        <td>{campaign['materials_executed']}</td>
                        <td>{campaign['materials_successful']}</td>
                        <td>{campaign['materials_failed']}</td>
                        <td>{campaign['success_rate']:.1f}%</td>
                        <td>{campaign['total_execution_time']:.0f}s</td>
                        <td><span class="status-badge {status_class}">{status_text}</span></td>
                    </tr>
        """
    
    dashboard_html += f"""
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Datos del dashboard
        const dashboardData = {json.dumps(dashboard_data)};
        
        // Configuración de gráficos
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.color = '#2c3e50';
        
        // Gráfico de tasa de éxito
        const successCtx = document.getElementById('successRateChart').getContext('2d');
        new Chart(successCtx, {{
            type: 'line',
            data: {{
                labels: dashboardData.campaigns.map(c => `Campaña ${{c.id}}`),
                datasets: [{{
                    label: 'Tasa de Éxito (%)',
                    data: dashboardData.campaigns.map(c => c.success_rate),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // Gráfico de materiales
        const materialsCtx = document.getElementById('materialsChart').getContext('2d');
        new Chart(materialsCtx, {{
            type: 'bar',
            data: {{
                labels: dashboardData.campaigns.map(c => `Campaña ${{c.id}}`),
                datasets: [
                    {{
                        label: 'Exitosos',
                        data: dashboardData.campaigns.map(c => c.materials_successful),
                        backgroundColor: '#27ae60',
                    }},
                    {{
                        label: 'Fallidos',
                        data: dashboardData.campaigns.map(c => c.materials_failed),
                        backgroundColor: '#e74c3c',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        stacked: true,
                    }},
                    y: {{
                        stacked: true,
                        beginAtZero: true
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                    }}
                }}
            }}
        }});
        
        // Actualización automática cada 30 segundos
        function updateDashboard() {{
            // Aquí se podría agregar lógica para actualizar datos en tiempo real
            console.log('Dashboard actualizado:', new Date().toLocaleTimeString());
        }}
        
        setInterval(updateDashboard, 30000);
        
        console.log('Dashboard inicializado correctamente');
    </script>
</body>
</html>
    """
    
    # Guardar dashboard
    dashboard_path = output_dir / "dashboard_interactivo.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"   ✅ Dashboard guardado: {dashboard_path}")
    return dashboard_path


def generar_reporte_pdf_resumen(campaign_results: List[CampaignResult], output_dir: Path):
    """Genera reporte PDF de resumen para publicación."""
    print("📄 GENERANDO REPORTE PDF")
    print("=" * 30)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Por simplicidad, generamos un reporte en texto que podría convertirse a PDF
    # En una implementación real, se usaría una librería como reportlab o weasyprint
    
    report_content = f"""
REPORTE TÉCNICO - SISTEMA DE PRECONVERGENCIA MULTIMATERIAL
===========================================================

Fecha de generación: {datetime.now().strftime("%d de %B de %Y")}
Hora: {datetime.now().strftime("%H:%M:%S")}

RESUMEN EJECUTIVO
-----------------
Este reporte presenta los resultados del análisis sistemático de 
preconvergencia DFT realizado sobre múltiples materiales semiconductores
utilizando el sistema de preconvergencia multimaterial desarrollado.

METODOLOGÍA
-----------
• Generación automática de combinaciones III-V y II-VI
• Filtrado por compatibilidad química
• Cálculos DFT de preconvergencia paralelos
• Análisis estadístico de parámetros convergidos

RESULTADOS PRINCIPALES
---------------------
• Campañas ejecutadas: {len(campaign_results)}
• Total de materiales analizados: {sum(cr.materials_executed for cr in campaign_results)}
• Materiales exitosos: {sum(cr.materials_successful for cr in campaign_results)}
• Tasa de éxito general: {sum(cr.success_rate for cr in campaign_results) / len(campaign_results) if campaign_results else 0:.1f}%

DETALLE POR CAMPAÑA
------------------
"""
    
    for i, campaign in enumerate(campaign_results, 1):
        report_content += f"""
Campaña {i}:
  - Materiales procesados: {campaign.materials_executed}
  - Análisis exitosos: {campaign.materials_successful}
  - Análisis fallidos: {campaign.materials_failed}
  - Tasa de éxito: {campaign.success_rate:.1f}%
  - Tiempo total: {campaign.total_execution_time/60:.1f} minutos
  - Tiempo promedio por material: {campaign.average_execution_time:.1f} segundos
"""
    
    report_content += """
CONCLUSIONES
------------
1. El sistema de preconvergencia multimaterial demuestra alta eficiencia
   en el procesamiento paralelo de múltiples semiconductores.

2. La tasa de éxito general indica robustez del sistema de convergencia.

3. Los tiempos de procesamiento son consistentes y escalables.

RECOMENDACIONES
---------------
• Continuar optimizando parámetros de convergencia para materiales específicos
• Implementar mejoras en el manejo de errores para reducir fallos
• Expandir la base de datos de semiconductores con más materiales experimentales

REFERENCIAS
-----------
• Sistema de Preconvergencia Multimaterial v2.0
• Documentación técnica completa disponible en /docs/
• Ejemplos de uso en /examples/

---
Reporte generado automáticamente por el Sistema de Preconvergencia Multimaterial
Contacto: research@preconvergencia.org
"""
    
    # Guardar reporte de texto (que puede convertirse a PDF)
    report_path = output_dir / "reporte_tecnico.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   ✅ Reporte técnico guardado: {report_path}")
    return report_path


def exportar_datos_multiples_formatos(campaign_results: List[CampaignResult], output_dir: Path):
    """Exporta datos en múltiples formatos."""
    print("💾 EXPORTANDO DATOS EN MÚLTIPLES FORMATOS")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Consolidar todos los datos
    all_data = []
    for i, campaign in enumerate(campaign_results):
        for result in campaign.individual_results:
            data_point = {
                'campaign_id': i + 1,
                'formula': result.formula,
                'success': result.success,
                'execution_time': result.execution_time,
                'stages_completed': ', '.join(result.stages_completed),
                'optimal_cutoff': result.optimal_cutoff,
                'optimal_kmesh': str(result.optimal_kmesh) if result.optimal_kmesh else None,
                'optimal_lattice_constant': result.optimal_lattice_constant
            }
            all_data.append(data_point)
    
    # 1. Exportar como CSV
    import pandas as pd
    df = pd.DataFrame(all_data)
    csv_path = output_dir / "resultados_completos.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV exportado: {csv_path}")
    
    # 2. Exportar como JSON
    json_path = output_dir / "resultados_completos.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    print(f"   ✅ JSON exportado: {json_path}")
    
    # 3. Exportar como YAML
    import yaml
    yaml_path = output_dir / "resultados_completos.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_data, f, default_flow_style=False, allow_unicode=True)
    print(f"   ✅ YAML exportado: {yaml_path}")
    
    # 4. Crear archivo de metadatos
    metadata = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'total_campaigns': len(campaign_results),
            'total_records': len(all_data),
            'successful_records': sum(1 for d in all_data if d['success']),
            'failed_records': sum(1 for d in all_data if not d['success'])
        },
        'data_summary': {
            'campaigns': [f"Campaña {i+1}" for i in range(len(campaign_results))],
            'materials_analyzed': list(set(d['formula'] for d in all_data)),
            'date_range': {
                'start': datetime.now().isoformat(),
                'end': datetime.now().isoformat()
            }
        },
        'file_locations': {
            'csv': str(csv_path),
            'json': str(json_path),
            'yaml': str(yaml_path)
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadatos guardados: {metadata_path}")
    
    return {
        'csv': csv_path,
        'json': json_path,
        'yaml': yaml_path,
        'metadata': metadata_path
    }


def generar_recomendaciones_automaticas(campaign_results: List[CampaignResult]) -> List[str]:
    """Genera recomendaciones automáticas basadas en los resultados."""
    recommendations = []
    
    if not campaign_results:
        return ["No hay datos suficientes para generar recomendaciones."]
    
    # Análisis de tasas de éxito
    success_rates = [cr.success_rate for cr in campaign_results]
    avg_success = sum(success_rates) / len(success_rates)
    
    if avg_success >= 85:
        recommendations.append("Excelente tasa de éxito general - El sistema está funcionando óptimamente")
    elif avg_success >= 70:
        recommendations.append("Buena tasa de éxito - Considerar optimizar materiales con fallos recurrentes")
    else:
        recommendations.append("Tasa de éxito mejorable - Revisar configuración de parámetros y materiales problemáticos")
    
    # Análisis de tiempos de ejecución
    avg_times = [cr.average_execution_time for cr in campaign_results]
    overall_avg_time = sum(avg_times) / len(avg_times)
    
    if overall_avg_time <= 120:
        recommendations.append("Tiempos de ejecución eficientes - El sistema está bien optimizado")
    elif overall_avg_time <= 300:
        recommendations.append("Tiempos de ejecución aceptables - Considerar optimización adicional")
    else:
        recommendations.append("Tiempos de ejecución altos - Revisar configuración de paralelización")
    
    # Análisis de consistencia
    time_std = (sum((t - overall_avg_time)**2 for t in avg_times) / len(avg_times))**0.5
    if time_std <= 30:
        recommendations.append("Alta consistencia en tiempos - Sistema estable y predecible")
    else:
        recommendations.append("Variabilidad en tiempos - Investigar cuellos de botella")
    
    # Recomendaciones específicas por campaña
    for i, campaign in enumerate(campaign_results, 1):
        if campaign.success_rate < 60:
            recommendations.append(f"Campaña {i}: Revisar materiales o configuración (éxito: {campaign.success_rate:.1f}%)")
        if campaign.average_execution_time > 300:
            recommendations.append(f"Campaña {i}: Optimizar paralelización (tiempo: {campaign.average_execution_time:.0f}s)")
    
    return recommendations


async def ejecutar_estudio_ejemplo():
    """Ejecuta un estudio de ejemplo para generar reportes."""
    print("🚀 EJECUTANDO ESTUDIO DE EJEMPLO")
    print("=" * 40)
    
    # Materiales para el estudio
    materiales_estudio = ['GaAs', 'GaN', 'InP', 'AlAs', 'ZnS', 'ZnSe', 'CdS', 'CdTe']
    
    # Crear múltiples campañas para simular estudios
    campaign_results = []
    
    for i in range(3):  # Simular 3 campañas
        print(f"🔄 Ejecutando campaña {i+1}/3...")
        
        # Simular campaña con resultados variables
        campaign = crear_campana_simulada(f"Estudio_{i+1}", materiales_estudio)
        campaign_results.append(campaign)
        
        print(f"   ✅ Completada: {campaign.materials_successful}/{campaign.materials_executed} exitosos")
    
    return campaign_results


def crear_campana_simulada(nombre: str, materiales: List[str]):
    """Crea una campaña simulada para demostración."""
    import random
    
    # Simular resultados variables por campaña
    resultados = []
    
    for material in materiales:
        # Variabilidad entre campañas
        success_prob = 0.7 + 0.1 * random.random()  # 70-80%
        success = random.random() < success_prob
        
        execution_time = 60 + 30 * random.random()  # 60-90s
        
        if success:
            optimal_cutoff = 400 + 50 * random.random()  # 400-450 Ry
            optimal_lattice = 5.5 + 0.2 * random.random()  # 5.5-5.7 Å
            stages = ['cutoff', 'kmesh', 'lattice']
        else:
            optimal_cutoff = None
            optimal_lattice = None
            stages = ['cutoff'] if random.random() < 0.5 else []
        
        from workflow.multi_material_pipeline import MaterialExecutionResult
        
        resultado = MaterialExecutionResult(
            formula=material,
            success=success,
            execution_time=execution_time,
            stages_completed=stages,
            optimal_cutoff=optimal_cutoff,
            optimal_kmesh=(6, 6, 6) if success else None,
            optimal_lattice_constant=optimal_lattice if success else None
        )
        resultados.append(resultado)
    
    # Calcular estadísticas de la campaña
    successful = [r for r in resultados if r.success]
    failed = [r for r in resultados if not r.success]
    
    from workflow.multi_material_pipeline import CampaignResult
    from core.multi_material_config import MultiMaterialConfig
    
    campaign = CampaignResult(
        materials_executed=len(materiales),
        materials_successful=len(successful),
        materials_failed=len(failed),
        total_execution_time=sum(r.execution_time for r in resultados),
        individual_results=resultados,
        campaign_config=MultiMaterialConfig()
    )
    
    return campaign


async def main():
    """Función principal del generador de reportes."""
    print("📊 GENERADOR DE REPORTES AUTOMÁTICOS")
    print("=" * 50)
    
    # 1. Ejecutar estudio de ejemplo
    campaign_results = await ejecutar_estudio_ejemplo()
    
    # 2. Crear directorio de salida
    output_dir = Path("results/reportes_automaticos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Generar reportes
    print(f"\n📋 GENERANDO REPORTES")
    print("=" * 25)
    
    # Reporte ejecutivo HTML
    html_path = generar_html_ejecutivo(campaign_results, output_dir)
    
    # Dashboard interactivo
    dashboard_path = generar_dashboard_interactivo(campaign_results, output_dir)
    
    # Reporte PDF (texto)
    pdf_path = generar_reporte_pdf_resumen(campaign_results, output_dir)
    
    # Exportación de datos
    export_paths = exportar_datos_multiples_formatos(campaign_results, output_dir)
    
    # 4. Crear índice de reportes
    print(f"\n📁 CREANDO ÍNDICE DE REPORTES")
    print("=" * 35)
    
    index_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Índice de Reportes - Preconvergencia Multimaterial</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .reports-grid {{ display: grid; gap: 20px; }}
        .report-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; }}
        .report-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
        .report-desc {{ margin-bottom: 15px; color: #666; }}
        .report-link {{ background: #3498db; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
        .report-link:hover {{ background: #2980b9; }}
        .timestamp {{ color: #888; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Centro de Reportes</h1>
        <p>Sistema de Preconvergencia Multimaterial</p>
    </div>
    
    <div class="reports-grid">
        <div class="report-card">
            <div class="report-title">📈 Reporte Ejecutivo HTML</div>
            <div class="report-desc">Reporte visual comprensivo con métricas principales y recomendaciones automáticas</div>
            <a href="reporte_ejecutivo.html" class="report-link">Ver Reporte</a>
        </div>
        
        <div class="report-card">
            <div class="report-title">📊 Dashboard Interactivo</div>
            <div class="report-desc">Dashboard en tiempo real con gráficos dinámicos y tabla de campañas</div>
            <a href="dashboard_interactivo.html" class="report-link">Ver Dashboard</a>
        </div>
        
        <div class="report-card">
            <div class="report-title">📄 Reporte Técnico PDF</div>
            <div class="report-desc">Reporte técnico detallado para publicación y documentación científica</div>
            <a href="reporte_tecnico.txt" class="report-link">Descargar PDF</a>
        </div>
        
        <div class="report-card">
            <div class="report-title">💾 Datos de Resultados</div>
            <div class="report-desc">Datos completos exportados en múltiples formatos (CSV, JSON, YAML)</div>
            <div style="margin-top: 10px;">
                <a href="resultados_completos.csv" class="report-link" style="margin-right: 10px;">CSV</a>
                <a href="resultados_completos.json" class="report-link" style="margin-right: 10px;">JSON</a>
                <a href="resultados_completos.yaml" class="report-link">YAML</a>
            </div>
        </div>
    </div>
    
    <div class="timestamp">
        Generado automáticamente el {datetime.now().strftime("%d de %B de %Y a las %H:%M")}
    </div>
</body>
</html>
    """
    
    index_path = output_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"   ✅ Índice de reportes: {index_path}")
    
    # 5. Resumen final
    print(f"\n🎉 GENERACIÓN DE REPORTES COMPLETADA")
    print("=" * 45)
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"📊 Reportes generados:")
    print(f"   • Reporte Ejecutivo: reporte_ejecutivo.html")
    print(f"   • Dashboard Interactivo: dashboard_interactivo.html")
    print(f"   • Reporte Técnico: reporte_tecnico.txt")
    print(f"   • Datos CSV: resultados_completos.csv")
    print(f"   • Datos JSON: resultados_completos.json")
    print(f"   • Datos YAML: resultados_completos.yaml")
    print(f"   • Índice principal: index.html")
    
    print(f"\n🌐 Para ver los reportes:")
    print(f"   Abra: file://{index_path.absolute()}")
    
    return output_dir


if __name__ == "__main__":
    try:
        results_dir = asyncio.run(main())
        print(f"\n✅ Generador de reportes completado exitosamente")
    except KeyboardInterrupt:
        print(f"\n⏹️  Generación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la generación: {e}")
        import traceback
        traceback.print_exc()