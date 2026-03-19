# magnet-design-MIE1603

The AMNS matrix is tracked using Git LFS. After cloning the project from GitHub, set up Git LFS as instructed [here](https://git-lfs.com) if you have not already. 
- Run `git lfs install` in your repository 
- Run `git lfs pull` to retrieve the AMNS matrix 


All the useful stuff is in the `magnet_design` directory. 

Order to find stuff: 
- `magnet_design/optimization/model.py` 
- `magnet_design/optimization/constraints.py` 
    - For each constraint class, the main content is in the `generate()` function 


Difference in variable names: 
- `x_vars` in original repo is `vars_i` in jupyter notebook 
- `y_vars` in original repo is `vars_x` in jupyter notebook 
- When you use all the variables in constraints, always use `array_i` and `array_x` for easier array manipulation 
