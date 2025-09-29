import pyvista as pv
import numpy as np
import sys

from vtk import (
    VTK_VERTEX, VTK_POLY_VERTEX, VTK_LINE, VTK_POLY_LINE,
    VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON, VTK_TETRA,
    VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID
)

VTK_CELL_TYPES = {
    VTK_VERTEX: "Vertex",
    VTK_POLY_VERTEX: "PolyVertex",
    VTK_LINE: "Line",
    VTK_POLY_LINE: "PolyLine",
    VTK_TRIANGLE: "Triangle",
    VTK_QUAD: "Quad",
    VTK_POLYGON: "Polygon",
    VTK_TETRA: "Tetrahedron",
    VTK_HEXAHEDRON: "Hexahedron",
    VTK_WEDGE: "Wedge (Prism)",
    VTK_PYRAMID: "Pyramid",
}

def summarize_vtk(filename):
    mesh = pv.read(filename)

    print(f"\n📂 File: {filename}")
    print(f"📌 Dataset type: {mesh.__class__.__name__}")
    print(f"🔹 Number of points: {mesh.n_points}")
    print(f"🔹 Number of cells: {mesh.n_cells}")

    if mesh.n_cells == 0 and mesh.n_points > 0:
        print("⚪ Looks like a point cloud")
    elif mesh.n_cells > 0 and mesh.n_points > 0:
        if mesh.is_all_triangles:
            print("🔺 Looks like a surface mesh (triangles)")
        else:
            print("🧊 Looks like a volumetric/unstructured grid")

    # ✅ Cell type summary
    if hasattr(mesh, "celltypes"):  # UnstructuredGrid
        unique_types, counts = np.unique(mesh.celltypes, return_counts=True)
        print("\n📦 Cell Types:")
        for t, c in zip(unique_types, counts):
            name = VTK_CELL_TYPES.get(t, f"Unknown({t})")
            print(f"  - {name:<15} (VTK code {t}): {c}")
    else:
        # Fallback for PolyData – just show how many cells of each dimension
        cell_sizes = [cell.n_points for cell in mesh.cells]
        print("\n📦 Cell info (PolyData):")
        print(f"  - Min cell size: {min(cell_sizes)} points")
        print(f"  - Max cell size: {max(cell_sizes)} points")
        print(f"  - Avg cell size: {np.mean(cell_sizes):.2f} points")

    # Point data
    if mesh.point_data:
        print("\n📊 Point Data Fields:")
        for name, arr in mesh.point_data.items():
            print(f"  - {name}: shape={arr.shape}, "
                  f"min={np.min(arr):.4g}, max={np.max(arr):.4g}, mean={np.mean(arr):.4g}")

    # Cell data
    if mesh.cell_data:
        print("\n📊 Cell Data Fields:")
        for name, arr in mesh.cell_data.items():
            print(f"  - {name}: shape={arr.shape}, "
                  f"min={np.min(arr):.4g}, max={np.max(arr):.4g}, mean={np.mean(arr):.4g}")

    if not mesh.point_data and not mesh.cell_data:
        print("\nℹ️ No field data found (geometry only).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vtk_examine.py <file.vtk>")
    else:
        summarize_vtk(sys.argv[1])
