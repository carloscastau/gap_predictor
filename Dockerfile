FROM python:3.10-slim

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYSCF_MAX_MEMORY=8000

# Instalar dependencias del sistema para PySCF y bibliotecas científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran cmake \
    libopenblas-dev liblapack-dev libfftw3-dev libhdf5-dev \
    libxc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos del proyecto
COPY requirements.txt .
COPY preconvergencia_GaAs.py .
COPY optimization_guide.md .
COPY diagnostics.py .
COPY advanced_optimization.py .
COPY optimization_config.json .
COPY test_docker.py .
COPY resume_checkpoint.py .

# Instalar dependencias Python
RUN python -m pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade numpy scipy matplotlib pandas pymatgen spglib psutil

# Crear usuario no-root para seguridad
RUN useradd -m -s /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Configuración para desarrollo flexible
CMD ["python", "preconvergencia_GaAs.py", "--help"]
