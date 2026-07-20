#!/bin/bash
#SBATCH --job-name=sphere-sar
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
rm -f particle.out *.vtk
rm -rf input_files
mkdir -p input_files

# ---------- Parameters (match sphere_saramito.f90 namelist) ----------
meshfile='mesh.out'

# Fluid / model parameters
eta_s=0.5
Gmod=9.0
lambda=0.1
model=2           # 2: Oldroyd-B, 3: Giesekus, 5/6: PTT lin/exp
alam_model=0       # 0:none 1:elastic 2:Saramito1 3:Saramito2
alphapar=0.1       # Giesekus alpha
epspar=0.1         # PTT epsilon
tau_y=2.0
K=1.0
n=0.5

# Time stepping
deltat=0.005
numtimesteps=3000
vtkevery=100

# Forcing
force=25.132741
t0=0.2             # creep test: force applied for t<=t0, then 0

# Write input file
input_file="input_files/input_saramito.txt"
cat > "$input_file" << EOF
&comppar
  meshfile     = '$meshfile'
  eta_s        = $eta_s
  Gmod         = $Gmod
  lambda       = $lambda
  model        = $model
  alam_model   = $alam_model
  alphapar     = $alphapar
  epspar       = $epspar
  tau_y        = $tau_y
  K            = $K
  n            = $n
  deltat       = $deltat
  force        = $force
  t0           = $t0
  vtkevery     = $vtkevery
  numtimesteps = $numtimesteps
/
EOF

echo "==== Running sphere_saramito ===="
./sphere_saramito < "$input_file"
echo "==== Simulation completed. ===="
