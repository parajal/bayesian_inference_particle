#!/bin/bash

#SBATCH --job-name=particle8p
#SBATCH --output=blob-full-8p-%j.out
#SBATCH --partition=mech-cem.cpu.q
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

set -euo pipefail

# ======================================================================
#  Run setup (directories, environment, build)
# ======================================================================
# Under sbatch the script is copied to a spool dir, so BASH_SOURCE points there.
# SLURM_SUBMIT_DIR is the directory sbatch was launched from (the project dir).
source_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source_dir="$(cd "$source_dir" && pwd)"
run_dir="${RUN_DIR:-$source_dir}"
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"

module load intel

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_DYNAMIC=FALSE

executable="magnetic_viscoelastic_pardiso_full_8p"
run_tag="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "==== Building $executable ===="
make -C "$source_dir" "$executable"

if [[ "$run_dir" != "$source_dir" ]]; then
  cp "$source_dir/full_particle_triperiodic.igo" "$run_dir/"
  cp "$source_dir/refinement.igo" "$run_dir/"
fi

# ======================================================================
#  INPUT PARAMETERS  (edit below)
# ======================================================================

# ---------- Fluid properties ----------
eta_s=0.5
eta_p=0.9
lambda=0.1

# ---------- Gaussian G profile ----------
base_G=9.0
n_blobs=20
seed=42
shift_G=.true.      # .true. only for the old rigidly-translated-field workflow
shift_seed=0
compact_gaussian=.false.
overlap_blobs=.true.
possible_blob_radii="0.5, 1.0, 1.5, 2.0"
possible_blob_amplitudes="2.0, 5.0, 10.0, 20.0"

# ---------- Particle placement ----------
# Provide your own particle centres here. The number of particles is derived
# from the number of values you list: e.g. x_center="5.0" -> 1 particle,
# x_center="3.0, 7.0" -> 2 particles. X/Y/Z must have the same count.
# Leave all three empty ("") to fall back to random placement of n_part_random
# particles using particle_seed.
x_center="0"
y_center="0"
z_center="0"

n_part_random=10         # particles to place randomly when no centres are given
particle_seed=12345      # seed for random placement (integer, or "None")

particle_min_dist=1.0          # min surface gap, in radii
particle_wall_min_dist=0.0     # optional wall clearance, in radii
particle_wall_min_elems=2.0    # min elements in close wall gaps

# ---------- Transported phi-to-G mapping ----------
phi_blob=1.0
phi_bg=1.0
phi_ref=1.0
phi_exp=1.0
lambda_exp=0.0
G_min=1.0e-12

# ---------- Constitutive model ----------
model=2
alam_model=0
logc=1
tau_y=0.00

# ---------- Numerical parameters ----------
dx_box=0.7
dx_part=0.35
particle_transition_elems=20.0
particle_blob_mesh_ratio=1.5
uniform_mesh=.false.
refine_blobs=.false.
refine_all_blobs=.false.
blob_refine_distance=-1.0
blob_mesh_factor=4.0
deltat=0.01
force1=25.132741
time_stop=0.2            # creep recovery: force off after this time

# ---------- Domain dimensions ----------
lx=10.0
ly=10.0
lz=10.0

# ---------- Output ----------
vtkevery=5
numtimesteps=50

# ======================================================================
#  Derive particle count / centre namelist lines
# ======================================================================
clean_center_list() {
  local values="$1"
  values="${values#[}"
  values="${values%]}"
  values="${values//;/,}"
  echo "$values"
}

count_center_values() {
  local values
  values="$(clean_center_list "$1")"
  values="${values//,/ }"
  set -- $values
  echo "$#"
}

center_nml_lines=""
if [[ -n "$x_center" || -n "$y_center" || -n "$z_center" ]]; then
  if [[ -z "$x_center" || -z "$y_center" || -z "$z_center" ]]; then
    echo "x_center, y_center, and z_center must all be set together." >&2
    exit 1
  fi
  x_center="$(clean_center_list "$x_center")"
  y_center="$(clean_center_list "$y_center")"
  z_center="$(clean_center_list "$z_center")"
  nx_center="$(count_center_values "$x_center")"
  ny_center="$(count_center_values "$y_center")"
  nz_center="$(count_center_values "$z_center")"
  if [[ "$nx_center" -eq 0 || "$nx_center" -ne "$ny_center" || "$nx_center" -ne "$nz_center" ]]; then
    echo "x/y/z_center must contain the same nonzero number of values." >&2
    echo "Counts: X=$nx_center Y=$ny_center Z=$nz_center" >&2
    exit 1
  fi
  n_part="$nx_center"   # number of particles derived from the centre list
  center_nml_lines="  x_center = $x_center,
  y_center = $y_center,
  z_center = $z_center,"
else
  n_part="$n_part_random"
fi

# ---------- Particle placement seed (only used for random placement) ----------
if [[ "${particle_seed,,}" == "none" ]]; then
  particle_seed_nml=-1
elif [[ "$particle_seed" =~ ^-?[0-9]+$ ]]; then
  particle_seed_nml="$particle_seed"
else
  echo "Invalid particle_seed='$particle_seed': use an integer or None." >&2
  exit 1
fi

# ---------- Case id ----------
if [[ -n "${CASE_ID:-}" ]]; then
  case_id="$CASE_ID"
elif [[ -n "$center_nml_lines" ]]; then
  case_id="full_triperiodic_${n_part}p_centers_${run_tag}"
elif [[ "${shift_G,,}" == ".true." || "${shift_G,,}" == "true" ]]; then
  case_id="full_triperiodic_${n_part}p_shift${shift_seed}_${run_tag}"
else
  case_id="full_triperiodic_${n_part}p_fixedG_${run_tag}"
fi

# ======================================================================
#  Write namelist input file
# ======================================================================
cd "$run_dir"
mkdir -p input_files

input_file="input_files/input_viscoelastic_${case_id}.txt"
cat > "$input_file" << EOF
&comppar
  eta_s    = $eta_s,
  eta_p    = $eta_p,
  lambda   = $lambda,

  base_G        = $base_G,
  n_blobs       = $n_blobs,
  seed          = $seed,
  particle_seed = $particle_seed_nml,
$center_nml_lines
  shift_G       = $shift_G,
  shift_seed    = $shift_seed,
  compact_gaussian = $compact_gaussian,
  overlap_blobs    = $overlap_blobs,
  possible_blob_radii      = $possible_blob_radii,
  possible_blob_amplitudes = $possible_blob_amplitudes,

  n_part                 = $n_part,
  particle_min_dist      = $particle_min_dist,
  particle_wall_min_dist = $particle_wall_min_dist,
  particle_wall_min_elems = $particle_wall_min_elems,

  phi_blob      = $phi_blob,
  phi_bg        = $phi_bg,
  phi_ref       = $phi_ref,
  phi_exp       = $phi_exp,
  lambda_exp    = $lambda_exp,
  G_min         = $G_min,

  model      = $model,
  alam_model = $alam_model,
  logc       = $logc,
  tau_y      = $tau_y,

  dx_box    = $dx_box,
  dx_part   = $dx_part,
  particle_transition_elems = $particle_transition_elems,
  particle_blob_mesh_ratio = $particle_blob_mesh_ratio,
  uniform_mesh = $uniform_mesh,
  refine_blobs = $refine_blobs,
  refine_all_blobs = $refine_all_blobs,
  blob_refine_distance = $blob_refine_distance,
  blob_mesh_factor = $blob_mesh_factor,
  deltat    = $deltat,
  force1    = $force1,
  time_stop = $time_stop,

  lx = $lx,
  ly = $ly,
  lz = $lz,

  vtkevery     = $vtkevery,
  numtimesteps = $numtimesteps
/
EOF

# ======================================================================
#  Run
# ======================================================================
run_log="run_blob_${case_id}.log"
echo "==== Running $n_part-particle triperiodic fixed-G case ===="
"$source_dir/$executable" < "$input_file" 2>&1 | tee "$run_log"
if grep -Eq 'ERROR:|Error in (add_to_mesh|MA57|PARDISO)|(U|Omega) =.*\*' "$run_log"; then
  echo "==== Simulation failed: placement/solver error. See $run_log. ===="
  exit 1
fi
echo "==== Multi-particle simulation completed in $run_dir. Trajectories: particle_1.._${n_part}_trajectory_full.out ===="
