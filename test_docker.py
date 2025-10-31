#!/usr/bin/env python3
"""
Script simple de prueba para verificar que el ambiente Docker funciona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf import gto
from pyscf.pbc import gto as pbc_gto, dft
import sys

print("🐳 PRUEBA DEL AMBIENTE DOCKER")
print("=" * 40)

try:
    # Test 1: Librerías básicas
    print("✅ Test 1: Librerías básicas")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")

    # Test 2: PySCF básico
    print("✅ Test 2: PySCF básico")
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.build()
    print(f"   Molécula H2: {mol.natm} átomos, {mol.nelectron} electrones")

    # Test 3: PySCF PBC (para sólidos)
    print("✅ Test 3: PySCF PBC")
    cell = pbc_gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.0"
    cell.a = np.eye(3) * 2.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.build()
    print(f"   Celda PBC: {cell.natm} átomos")

    # Test 4: DFT básico
    print("✅ Test 4: DFT básico")
    kpts = cell.make_kpts([2, 2, 2])
    kmf = dft.KRKS(cell, kpts=kpts)
    kmf.xc = "PBE"
    print(f"   DFT configurado: {len(kpts)} k-points")

    print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    print("El ambiente Docker está funcionando correctamente.")

except Exception as e:
    print(f"\n❌ Error durante las pruebas: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)