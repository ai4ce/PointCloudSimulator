This repository contains code for simulating a sequence of local 2D point clouds that are scanned from a pose trajectory in a virtual 2D environment

# Get Started
The simulation can be decomposed into three steps that generate
* 2D environment
* Pose trajectory
* Local point clouds

## Step 1: Creating 2D Environment
To create 2D environment, go to `./code` and run
```
python sample_maps.py --n_env 2 --img_h 1024 --img_w 1024
```
This command simulates 2 different environments and saves them in `../results/` with 3 sub-folders (`v0` and `v2`). In each sub-folder, there is a binary image `environment.png` that visualizes the simluated envrionment where the black pixels show the occupied locations and the white pixels are free-space.

## Step 2: Sampling Pose Trajectory
The pose trajectory is sampled **interactively** using the matlab script `script_samplePoseTrajectory.m`. You need to specify the environment name that is simulated from step 1 and also provide a name for this pose trajectory (because you can sample multiple trajectories within the same environment). 
```matlab
%% settings
env_idx = 'v0';
pose_idx = 'pose0';
```
The sampled pose will be saved as `../results/v0_pose0/gt_pose.mat`. 
When you run the script, a figure window will be created that displays the binary environment image. You need to select a polyline in the current figure using the mouse. This polyline is the base of the trajectory where the poses are sampled. 

Other parameters you can adjust
```matlab
n_pose_on_traj = 256;  % number of poses sampled on this trajectory
angle = 10; % max angular differences (in degree) two poses
displacement = 5; % controls the displacement of the sampled pose location from the polyline
```
## Step3: Scanning Local Point Cloud
To simulate 2D point clouds from the sampled pose trajectory in the 2D environment, run
```
python scan_local_point_clouds.py --data_dir ../results/ --env v0 --pose pose0 --obs 256
```
A set of local point clouds will be saved as `pcd` files in ```../results/v0/v0_pose0/```. The argument `--obs` controls the number of scanned points in each point cloud.

## File Structure
Once you finish the simulation, the `results` folder should look like (depending on how many environments and poses you simulated)
```
./results
├── v0
│   ├── environment.pkl
│   ├── environment.png
│   ├── v0_pose0
│   │   ├── 000.pcd
│   │   ├── 001.pcd
│   │   ├── ...
│   │   ├── ...
│   │   ├── gt_pose.mat
│   │   └── gt_pose.png
│   └── v0_pose1
│       ├── 000.pcd
│       ├── 001.pcd
│       ├── ...
│       ├── ...
│       ├── gt_pose.mat
│       └── gt_pose.png
├── v1
│   ├── environment.pkl
│   ├── environment.png
│   ├── v1_pose0
│   │   ├── 000.pcd
│   │   ├── 001.pcd
│   │   ├── ...
│   │   ├── ...
│   │   ├── gt_pose.mat
│   │   └── gt_pose.png
```
