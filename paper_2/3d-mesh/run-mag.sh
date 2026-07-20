#!/bin/bash

#SBATCH --job-name=stokes-0.6
#SBATCH --output=%j.out
#SBATCH --partition=mech-cem.cpu.q 
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G

# Load modules
module load intel
rm -f particle.txt

# Cleanup: remove old files and create input directory
rm -rf input_files
mkdir -p input_files

# Fixed parameters
eta=1.0
alpha1=1.0
alpha2=2.0
force1=37.6991
deltat=0.05
delta=1.0
angle=45.0
init_dist=4.0
nobj=1
rc=1.0
dx_box=3.0
dx_part=0.025                   
lx=200.0
ly=100.0
lz=200.0
nelem_min=3
H=10.0
magnetic=.false.
periodic=.false.
numtimesteps=120
vtkevery=5
volume_threshold=1.39
aspect_ratio_threshold=1.39

# Write single input file
input_file="input_files/input_stokes.txt"
cat > "$input_file" << EOF
&comppar
  eta = $eta,
  alpha1 = $alpha1,
  alpha2 = $alpha2,
  force1 = $force1,
  deltat = $deltat,
  delta = $delta,
  angle = $angle,
  init_dist = $init_dist,
  nobj = $nobj,
  rc = $rc,
  dx_box = $dx_box,
  dx_part = $dx_part,
  lx = $lx,
  ly = $ly,
  lz = $lz,
  nelem_min = $nelem_min,
  H = $H,
  magnetic = $magnetic,
  periodic = $periodic,
  numtimesteps = $numtimesteps,
  vtkevery = $vtkevery,
  volume_threshold = $volume_threshold,
  aspect_ratio_threshold = $aspect_ratio_threshold
/
EOF

echo "==== Running simulation with Upart = $upart ===="
./magnetic_stokes_quarter < "$input_file"
echo "==== Simulation completed. ===="
