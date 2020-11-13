## Error-state Kalman filter

Working ESKF which estimates a vehicles pose based on IMU-measurements and GNSS.

## Code:

run_INS_simulated.py runs the ESKF on a simulated dataset.

run_INS_real.py runs the ESKF on a real UAV. Quite large dataset, variables StartTime and N set the starting iteration and ending iteration, respectively. First 50 000 iterations are of the UAV standing still on the ground.

## How to run:

Run the chosen run-file in python 3.6 or higher. Recommended to run with -O flag.
