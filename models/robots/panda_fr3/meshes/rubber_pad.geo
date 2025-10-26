SetFactory("OpenCASCADE");

// Pad dimensions (edit these to match Panda pad size)
Box(1) = {0, 0, 0, 0.1, 0.1, 0.02}; // x, y, z, dx, dy, dz

// Mesh size control
Mesh.CharacteristicLengthMax = 0.005;

// Generate tetrahedra
Mesh 3;

