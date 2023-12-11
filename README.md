## Design Space Exploration Algorithm for Delta-Sigma Modulators

This directory contains the code for the high-level topological synthesis method for delta-sigma modulators. Following the following steps to run the code:

1. Before running the code, please make sure that the libraries "hebo" and "matlab.engine" are installed in Python (recommended 3.6.13) environment, as well as other common scientific computing libraries. The environment configuration can be found in the environment.yml file located in the root of this repository.

2. Please replace 'user' in the kill_matlab.sh script with your own username before running;

3. The main program is optimization.py, run the command
```bash
python -W ignore optimization.py
```

4. All modulator topologies and performance results can be viewed in the corresponding html format file under database;

5. Simulink models that satisfy the constraints are stored in the results folder;

6. A high-level model of a sample modulator (adc_test.slx) is given under the case_study path, which can be opened by simulink for viewing and simulation.

7. The high-level topology synthesis of delta-sigma modulator requires MATLAB and Simulink, MATLAB version 2017.b is recommended, in addition, SDToolbox2 toolbox needs to be installed for modulator modeling and simulation.
The toolbox can be found at https://www.mathworks.com/matlabcentral/fileexchange/25811-sdtoolbox-2/

## Reference 
Please cite the following paper, if this repo is used in your work. 

J. Lu, Y. Li, F. Yang, L. Shang and X. Zeng, "High-Level Topology Synthesis Method for Δ-Σ Modulators via Bi-Level Bayesian Optimization," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 70, no. 12, pp. 4389-4393, Dec. 2023, doi: 10.1109/TCSII.2023.3292389.
