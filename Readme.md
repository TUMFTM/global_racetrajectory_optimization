# Python version
Our code is tested with Python >= 3.7.4

# List of components
* `helper_funcs_glob`: This python module contains some helper functions used in several other functions when 
calculating the global race trajectory.
* `inputs`: This folder contains the ggv diagrams and reference track csvs for the global race trajectory.
* `int_globalenv_trajgen`: This python module contains a function to import the track csvs in the inputs folder.
* `params`: This folder contains the vehicle dependent parameter files.
* `process_functions`: This folder contains scripts to reduce the code contained in the `main_globtraj.py`.

# Dependencies
Use the provided `requirements.txt` in the root directory of this repo, in order to install all required modules.\
`pip3 install -r /path/to/requirements.txt`

# Running the code
* `Step 1:` (optional) Adjust the parameter file that can be found in the params folder.
* `Step 2:` (optional) Adjust the ggv diagram file in `inputs/ggv`.
* `Step 3:` (optional) Add your own reference track file in `inputs/tracks`.
* `Step 4:` Check the parameters in the upper part of `main_globaltraj.py` and execute it to start the trajectory
generation process. The calculated race trajectory can be found in `outputs/`.

![Resulting raceline for the Berlin FE track](opt_raceline_berlin.png)

# Wording and conventions
We tried to keep a consistant wording for the variable names:

path -> [x, y] Describes any array containing x,y coordinates of points (i.e. point coordinates).\
refline -> [x, y] A path that is used as reference line during our calculations.\
reftrack -> [x, y, w_tr_right, w_tr_left] An array that contains not only the reference line information but also right
and left track widths. In our case it contains the race track that is used as a basis for the raceline optimization.

Normal vectors usually point to the right in the direction of driving. Therefore, we get the track boundaries by
multiplication as follows: norm_vector * w_tr_right, -norm_vector * w_tr_left.

# Trajectory interface definition
The output csv contains the global race trajectory. The array is of size
[no_points x 7] where no_points depends on stepsize and track length. The seven columns are structured as follows:

* `s_m`: float32, meter. Curvi-linear distance along the raceline.
* `x_m`: float32, meter. X-coordinate of raceline point.
* `y_m`: float32, meter. Y-coordinate of raceline point.
* `psi_rad`: float32, rad. Heading of raceline in current point from -pi to +pi rad. Zero is north (along y-axis).
* `kappa_radpm`: float32, rad/meter. Curvature of raceline in current point.
* `vx_mps`: float32, meter/second. Target velocity in current point.
* `ax_mps2`: float32, meter/second². Target acceleration in current point.

# Detailed description of the curvature minimization used during trajectory generation
Please refer to our paper for further information:\
Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann\
Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar\
DOI: 10.1080/00423114.2019.1631455

Contact person: [Alexander Heilmeier](mailto:alexander.heilmeier@tum.de).
