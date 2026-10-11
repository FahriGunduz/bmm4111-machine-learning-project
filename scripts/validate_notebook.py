"""Defteri aktif Python ortamında yürüt; büyük çıktıları yalnız yerelde sakla."""
import argparse
import os
from pathlib import Path
import subprocess

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tmp/notebook")
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    import sys
    prefix = ROOT / ".tools" / "jupyter"
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(prefix), "--name", "bmm4111-local"], check=True)
    kernel_dir = prefix / "share" / "jupyter" / "kernels"
    kernel_manager = KernelManager(kernel_name="bmm4111-local", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    os.environ["BMM4111_OUTPUT_DIR"] = str(target)
    notebook = nbformat.read(ROOT / "notebooks/ML_Donem_Odevi.ipynb", as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, km=kernel_manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, target / "executed.ipynb")
    executed = sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells)
    print(f"PASS: {executed} kod hücresi yürütüldü. Çıktılı defter yalnız {target} içinde.")
