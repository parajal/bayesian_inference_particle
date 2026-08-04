#!/bin/bash
#SBATCH --job-name=creep-m3
#SBATCH --output=%j.out
#SBATCH --partition=mech-cem.cpu.q
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G

# Load required modules
module load intel

# Cleanup old files and prepare input directory
rm -f particle.txt
rm -rf input_files
mkdir -p input_files

# ---------- Fixed parameters ----------
eta_s=0.5
eta_p=0.9
alpha1=1.0
alpha2=2.0
lambda=0.1
mobility=0.2
model=2
alam_model=0
tau_y=0.01
delta=0.1
angle=45.0
init_dist=4.0
dx_box=24.0
nelem_min=3
dx_part=0.2
H=22.6274
deltat=0.005
force1=25.1327
time_stop=0.2
nobj=1
volume_threshold=1.39
lx=200.0
ly=100.0
lz=200.0
rc=1.0
aspect_ratio_threshold=1.39
periodic=.false.
magnetic=.false.
vtkevery=10
numtimesteps=4000

# Write single input file
input_file="input_files/input_viscoelastic.txt"
cat > "$input_file" << EOF
&comppar
  ! ---------- Fluid properties ----------
  eta_s    = $eta_s
  eta_p    = $eta_p
  alpha1   = $alpha1
  alpha2   = $alpha2
  lambda   = $lambda
  mobility = $mobility
  model    = $model
  alam_model = $alam_model
  tau_y = $tau_y

  ! ---------- Numerical parameters ----------
  delta    = $delta
  angle    = $angle
  init_dist= $init_dist
  dx_box   = $dx_box
  nelem_min= $nelem_min
  dx_part  = $dx_part
  H        = $H
  deltat   = $deltat
  force1   = $force1
  time_stop = $time_stop
  nobj     = $nobj
  volume_threshold = $volume_threshold

  ! ---------- Domain dimensions ----------
  lx = $lx
  ly = $ly
  lz = $lz
  rc = $rc

  ! ---------- Misc parameters ----------
  aspect_ratio_threshold = $aspect_ratio_threshold
  periodic = $periodic
  magnetic = $magnetic
  vtkevery = $vtkevery
  numtimesteps = $numtimesteps

/
EOF

echo "==== Running simulation with Upart = $upart ===="
./magnetic_viscoelastic_pardiso < "$input_file"
echo "==== Simulation completed. ===="
