#!/bin/bash
#SBATCH --job-name=sphere-newt
#SBATCH --output=%j.out
#SBATCH --partition=mech-cem.cpu.q
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G

# Load required modules
module load intel

# Cleanup old outputs and prepare input directory
rm -f particle.out particle.txt *.vtk
rm -rf input_files
mkdir -p input_files

# ---------- Parameters (match sphere_newtonian.f90 namelist) ----------
meshfile='mesh.out'
eta=1.0
force=37.6991
deltat=0.05
numtimesteps=120

# Write input file
input_file="input_files/input_newtonian.txt"
cat > "$input_file" << EOF
&comppar
  meshfile     = '$meshfile'
  eta          = $eta
  force        = $force
  deltat       = $deltat
  numtimesteps = $numtimesteps
/
EOF

echo "==== Running sphere_newtonian ===="
./sphere_newtonian < "$input_file"
echo "==== Simulation completed. ===="
