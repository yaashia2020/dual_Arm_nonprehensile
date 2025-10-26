import meshio

# Read the original Gmsh-generated vtk
mesh = meshio.read("rubber_pad.vtk")

# Keep only tetrahedral cells
tetra_cells = [c for c in mesh.cells if c.type == "tetra"]

# Create a new clean mesh
tetra_mesh = meshio.Mesh(points=mesh.points, cells=tetra_cells)

# Write it back

tetra_mesh.write("rubber_pad_clean.vtk", file_format="vtk51")



print("✅ Wrote cleaned file: rubber_pad_clean.vtk (only tetrahedra)")

