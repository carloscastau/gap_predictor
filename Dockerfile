# Dockerfile para el proyecto de preconvergencia GaAs refactorizado
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash preconvergence
USER preconvergence
WORKDIR /home/preconvergence

# Copiar archivos de configuración de Python
COPY --chown=preconvergence:preconvergence pyproject.toml setup.py requirements.txt ./

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY --chown=preconvergence:preconvergence src/ ./src/
COPY --chown=preconvergence:preconvergence scripts/ ./scripts/
COPY --chown=preconvergence:preconvergence config/ ./config/

# Crear directorios de salida
RUN mkdir -p results logs

# Variables de entorno para PySCF
ENV PYSCF_MAX_MEMORY=4096
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Punto de entrada
ENTRYPOINT ["python", "scripts/run_preconvergence.py"]

# Comando por defecto
CMD ["--config", "config/docker.yaml", "--output_dir", "results"]
