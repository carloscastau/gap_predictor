#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_preconvergence.py

Script principal para ejecutar el pipeline de preconvergencia refactorizado.
"""

import sys
import asyncio
from pathlib import Path

# Añadir src al path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Importar módulos directamente
from config import settings
from workflow import pipeline
from utils import logging

# Acceder a las funciones/clases
PreconvergenceConfig = settings.PreconvergenceConfig
get_default_config = settings.get_default_config
run_preconvergence_pipeline = pipeline.run_preconvergence_pipeline
setup_global_logging = logging.setup_global_logging


async def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de preconvergencia DFT refactorizado")
    parser.add_argument("--config", type=str, help="Archivo de configuración YAML")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directorio de salida")
    parser.add_argument("--resume", type=str, help="Reanudar desde checkpoint")
    parser.add_argument("--fast", action="store_true", help="Configuración rápida")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado")

    args = parser.parse_args()

    # Configurar logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_global_logging(level=log_level)

    # Cargar configuración
    if args.config:
        config = PreconvergenceConfig.load_from_file(Path(args.config))
    elif args.fast:
        from config.settings import get_fast_config
        config = get_fast_config()
    else:
        config = get_default_config()

    # Ajustar directorio de salida
    config.output_dir = Path(args.output_dir)

    logger.info("🚀 Iniciando pipeline de preconvergencia refactorizado")
    logger.info(f"Configuración: {config.to_dict()}")

    try:
        # Ejecutar pipeline
        result = await run_preconvergence_pipeline(config, args.resume)

        if result.success:
            logger.info("✅ Pipeline completado exitosamente")
            logger.info(f"Tiempo total: {result.total_duration:.2f}s")
        else:
            logger.error(f"❌ Pipeline falló: {result.error_message}")

        return result.success

    except Exception as e:
        logger.error(f"Error fatal en pipeline: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)