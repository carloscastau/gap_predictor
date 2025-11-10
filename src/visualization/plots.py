# src/visualization/plots.py
"""Generadores de gráficos para resultados DFT."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.gridspec as gridspec

from ..utils.logging import StructuredLogger


class ConvergencePlotter:
    """Generador de gráficos de convergencia."""

    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.colors = {
            'cutoff': '#1f77b4',      # azul
            'kmesh': '#ff7f0e',       # naranja
            'lattice': '#2ca02c',     # verde
            'convergence': '#d62728', # rojo
            'fit': '#9467bd'          # morado
        }
        self.logger = StructuredLogger("ConvergencePlotter")

    def plot_cutoff_convergence(self, cutoff_data: pd.DataFrame,
                               output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica convergencia vs cutoff."""
        if cutoff_data.empty:
            self.logger.warning("No cutoff data available for plotting")
            return self._create_empty_plot("Cutoff Convergence - No Data")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Datos válidos
        valid_data = cutoff_data.dropna(subset=['E_tot_Ha'])

        if not valid_data.empty:
            ax.plot(valid_data['ke_cutoff_Ry'], valid_data['E_tot_Ha'],
                   'o-', color=self.colors['cutoff'], markersize=8,
                   linewidth=2, label='Datos DFT')

            # Marcar energía mínima
            e_min = valid_data['E_tot_Ha'].min()
            cutoff_opt = valid_data.loc[valid_data['E_tot_Ha'].idxmin(), 'ke_cutoff_Ry']

            ax.axhline(e_min, color=self.colors['convergence'], linestyle='--',
                      alpha=0.7, label=f'E_min = {e_min:.6f} Ha')
            ax.axvline(cutoff_opt, color=self.colors['convergence'], linestyle=':',
                      alpha=0.7, label=f'Cutoff ópt = {cutoff_opt} Ry')

            # Ajuste polinomial si hay suficientes puntos
            if len(valid_data) >= 3:
                try:
                    coeffs = np.polyfit(valid_data['ke_cutoff_Ry'], valid_data['E_tot_Ha'], 2)
                    cutoff_range = np.linspace(valid_data['ke_cutoff_Ry'].min(),
                                              valid_data['ke_cutoff_Ry'].max(), 100)
                    e_fit = np.polyval(coeffs, cutoff_range)
                    ax.plot(cutoff_range, e_fit, '--', color=self.colors['fit'],
                           alpha=0.8, label='Ajuste cuadrático')
                except:
                    pass

        ax.set_xlabel('Cutoff (Ry)', fontsize=12)
        ax.set_ylabel('Energía Total (Ha)', fontsize=12)
        ax.set_title('Convergencia vs Cutoff', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Cutoff convergence plot saved to {output_path}")

        return fig

    def plot_kmesh_convergence(self, kmesh_data: pd.DataFrame,
                              output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica convergencia vs k-mesh."""
        if kmesh_data.empty:
            self.logger.warning("No k-mesh data available for plotting")
            return self._create_empty_plot("k-mesh Convergence - No Data")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Datos válidos
        valid_data = kmesh_data.dropna(subset=['E_tot_Ha'])

        if not valid_data.empty:
            ax.plot(valid_data['N_kpts'], valid_data['E_tot_Ha'],
                   's-', color=self.colors['kmesh'], markersize=8,
                   linewidth=2, label='Datos DFT')

            # Labels de k-mesh
            for _, row in valid_data.iterrows():
                k_str = f"{int(row['kx'])}x{int(row['ky'])}x{int(row['kz'])}"
                ax.annotate(k_str, (row['N_kpts'], row['E_tot_Ha']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            # Marcar energía mínima
            e_min = valid_data['E_tot_Ha'].min()
            nkpts_opt = valid_data.loc[valid_data['E_tot_Ha'].idxmin(), 'N_kpts']

            ax.axhline(e_min, color=self.colors['convergence'], linestyle='--',
                      alpha=0.7, label=f'E_min = {e_min:.6f} Ha')

        ax.set_xlabel('Número de puntos k', fontsize=12)
        ax.set_ylabel('Energía Total (Ha)', fontsize=12)
        ax.set_title('Convergencia vs k-mesh', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"k-mesh convergence plot saved to {output_path}")

        return fig

    def plot_lattice_optimization(self, lattice_data: pd.DataFrame,
                                 fit_results: Optional[Dict[str, Any]] = None,
                                 output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica optimización de parámetro de red."""
        if lattice_data.empty:
            self.logger.warning("No lattice data available for plotting")
            return self._create_empty_plot("Lattice Optimization - No Data")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Datos válidos
        valid_data = lattice_data.dropna(subset=['E_tot_Ha'])

        if not valid_data.empty:
            # Gráfica principal
            ax1.plot(valid_data['a_Ang'], valid_data['E_tot_Ha'],
                    '^-', color=self.colors['lattice'], markersize=8,
                    linewidth=2, label='Datos DFT')

            # Ajuste parabólico
            if fit_results and 'A' in fit_results:
                A = fit_results['A']
                a0 = fit_results.get('a0', valid_data['a_Ang'].mean())
                E0 = fit_results.get('E0', valid_data['E_tot_Ha'].min())

                a_range = np.linspace(valid_data['a_Ang'].min(), valid_data['a_Ang'].max(), 100)
                e_fit = E0 + A * (a_range - a0)**2

                ax1.plot(a_range, e_fit, '--', color=self.colors['fit'],
                        alpha=0.8, label=f'Ajuste parabólico (R²={fit_results.get("r2", 0):.3f})')

                # Marcar mínimo teórico
                a_opt = a0  # Para potencial parabólico centrado
                e_min = E0
                ax1.plot(a_opt, e_min, 'rx', markersize=12, markeredgewidth=2,
                        label=f'Mínimo teórico: a={a_opt:.4f} Å')

            ax1.set_xlabel('Parámetro de red (Å)', fontsize=12)
            ax1.set_ylabel('Energía Total (Ha)', fontsize=12)
            ax1.set_title('Optimización de Parámetro de Red', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Residuos del ajuste
            if fit_results and 'A' in fit_results:
                A = fit_results['A']
                a0 = fit_results.get('a0', valid_data['a_Ang'].mean())
                E0 = fit_results.get('E0', valid_data['E_tot_Ha'].min())

                e_fitted = E0 + A * (valid_data['a_Ang'].values - a0)**2
                residuals = valid_data['E_tot_Ha'].values - e_fitted

                ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
                ax2.plot(valid_data['a_Ang'], residuals, 'o-',
                        color=self.colors['convergence'], markersize=6)
                ax2.set_xlabel('Parámetro de red (Å)', fontsize=12)
                ax2.set_ylabel('Residuos (Ha)', fontsize=12)
                ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Sin ajuste disponible', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Lattice optimization plot saved to {output_path}")

        return fig

    def plot_convergence_overview(self, data_dict: Dict[str, pd.DataFrame],
                                 output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica general de convergencia."""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Cutoff
        ax1 = fig.add_subplot(gs[0, 0])
        if 'cutoff' in data_dict and not data_dict['cutoff'].empty:
            valid_data = data_dict['cutoff'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                ax1.plot(valid_data['ke_cutoff_Ry'], valid_data['E_tot_Ha'],
                        'o-', color=self.colors['cutoff'])
        ax1.set_title('Cutoff Convergence')
        ax1.grid(True, alpha=0.3)

        # k-mesh
        ax2 = fig.add_subplot(gs[0, 1])
        if 'kmesh' in data_dict and not data_dict['kmesh'].empty:
            valid_data = data_dict['kmesh'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                ax2.plot(valid_data['N_kpts'], valid_data['E_tot_Ha'],
                        's-', color=self.colors['kmesh'])
        ax2.set_title('k-mesh Convergence')
        ax2.grid(True, alpha=0.3)

        # Lattice
        ax3 = fig.add_subplot(gs[0, 2])
        if 'lattice' in data_dict and not data_dict['lattice'].empty:
            valid_data = data_dict['lattice'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                ax3.plot(valid_data['a_Ang'], valid_data['E_tot_Ha'],
                        '^-', color=self.colors['lattice'])
        ax3.set_title('Lattice Optimization')
        ax3.grid(True, alpha=0.3)

        # Panel de información
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        info_text = "RESUMEN DE CONVERGENCIA\n\n"

        # Análisis de cutoff
        if 'cutoff' in data_dict and not data_dict['cutoff'].empty:
            valid_data = data_dict['cutoff'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                e_range = valid_data['E_tot_Ha'].max() - valid_data['E_tot_Ha'].min()
                info_text += f"• Cutoff: {len(valid_data)} puntos, rango energético: {e_range:.2e} Ha\n"

        # Análisis de k-mesh
        if 'kmesh' in data_dict and not data_dict['kmesh'].empty:
            valid_data = data_dict['kmesh'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                e_range = valid_data['E_tot_Ha'].max() - valid_data['E_tot_Ha'].min()
                info_text += f"• k-mesh: {len(valid_data)} puntos, rango energético: {e_range:.2e} Ha\n"

        # Análisis de lattice
        if 'lattice' in data_dict and not data_dict['lattice'].empty:
            valid_data = data_dict['lattice'].dropna(subset=['E_tot_Ha'])
            if not valid_data.empty:
                e_range = valid_data['E_tot_Ha'].max() - valid_data['E_tot_Ha'].min()
                info_text += f"• Lattice: {len(valid_data)} puntos, rango energético: {e_range:.2e} Ha\n"

        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

        fig.suptitle('Análisis General de Convergencia', fontsize=16, fontweight='bold')

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Convergence overview plot saved to {output_path}")

        return fig

    def _create_empty_plot(self, title: str) -> plt.Figure:
        """Crea gráfica vacía para casos sin datos."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Sin datos disponibles', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig


class BandStructurePlotter:
    """Generador de gráficos de estructura de bandas."""

    def __init__(self):
        self.logger = StructuredLogger("BandStructurePlotter")

    def plot_band_structure(self, kpoints: np.ndarray, bands: np.ndarray,
                           fermi_level: Optional[float] = None,
                           kpath_labels: Optional[List[str]] = None,
                           output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica estructura de bandas."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convertir a eV
        bands_ev = bands * 27.211386245988

        # Distancia k-path
        k_distances = np.arange(len(kpoints))

        # Plot bandas
        for band_idx in range(bands.shape[1]):
            ax.plot(k_distances, bands_ev[:, band_idx], 'b-', alpha=0.7, linewidth=1)

        # Nivel de Fermi
        if fermi_level is not None:
            fermi_ev = fermi_level * 27.211386245988
            ax.axhline(fermi_ev, color='red', linestyle='--', alpha=0.8,
                      label=f'Fermi level: {fermi_ev:.3f} eV')

        # Labels de puntos altos simetría
        if kpath_labels:
            # Simplificado: etiquetas en extremos
            ax.set_xticks([0, len(kpoints)-1])
            ax.set_xticklabels([kpath_labels[0], kpath_labels[-1]])

        ax.set_xlabel('k-path', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('Band Structure', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Band structure plot saved to {output_path}")

        return fig


class DOSPlotter:
    """Generador de gráficos de densidad de estados."""

    def __init__(self):
        self.logger = StructuredLogger("DOSPlotter")

    def plot_dos(self, energies: np.ndarray, dos: np.ndarray,
                fermi_level: Optional[float] = None,
                output_path: Optional[Path] = None) -> plt.Figure:
        """Gráfica densidad de estados."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Convertir a eV
        energies_ev = energies * 27.211386245988

        ax.plot(energies_ev, dos, 'b-', linewidth=2, label='DOS')

        # Nivel de Fermi
        if fermi_level is not None:
            fermi_ev = fermi_level * 27.211386245988
            ax.axvline(fermi_ev, color='red', linestyle='--', alpha=0.8,
                      label=f'Fermi level: {fermi_ev:.3f} eV')

        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel('DOS (states/eV)', fontsize=12)
        ax.set_title('Density of States', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"DOS plot saved to {output_path}")

        return fig