# src/analysis/__init__.py
"""Módulos de análisis de resultados DFT."""

from .statistics import ConvergenceStatistics, PerformanceAnalyzer
from .fitting import ConvergenceFitter, ExtrapolationAnalyzer
from .validation import ResultValidator, ConsistencyChecker

__all__ = [
    'ConvergenceStatistics',
    'PerformanceAnalyzer',
    'ConvergenceFitter',
    'ExtrapolationAnalyzer',
    'ResultValidator',
    'ConsistencyChecker'
]