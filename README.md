# Replication code for Schwarzwald et al., _Large-scale stability and the Greater Horn of Africa long and short rains_

This code replicates the analysis and figures in Schwarzwald et al. (in revision). 

This code is to be used in conjunction with the replication data at [XX]. 

To run: 
1. Create a conda environment using the included `environment_gha_stability.yaml` file through `conda env create -f environment_gha_stability.yaml` or `mamba env create -f environment_gha_stability.yaml` (we recommend the latter for speed)
2. Update paths in `dir_list.csv`, to where you downloaded the replication data (set the path to `climate_raw` as `raw`, the path to `climate_proc` as `proc`, the path to `aux_data` as `aux`, and where you would like output figures as `figs` 
3. Step through `complete_run.ipynb`

Note that because of size constraints, only data needed to replicate main text figures were included in the replication data set. Some intermediate data (such as saturation specific humidity, or raw daily data for equatorial-wide maps such as Figures 10-11) have not been included, but can be regenerated using the replication data and the included code. If you'd like to have access to the rest of the data, including data needed to replicate figures in Supplementary Material, feel free to get in touch with the corresponding author (Kevin Schwarzwald, at kevin.schwarzwald@columbia.edu). 

Troubleshooting
- import errors involving ESMF can usually be solved by explicitly specifying the latest `xesmf` version in installations

