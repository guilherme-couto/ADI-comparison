import numpy as np

# Read reference solution (FE dt = 0.005)
ref = []
with open('./simulation-files/last-FE-0.005.txt', 'r') as f: 
    for line in f:
        ref.append(float(line))

# Dictionary of methods and dt
cases = {'FE': ['0.020', '0.040', '0.060', '0.080'],
         'ADI1': ['0.020', '0.040', '0.060', '0.080', '0.100', '0.200'],
         'ADI2': ['0.020', '0.040', '0.060', '0.080', '0.100', '0.200']}

# Open file for writing the error analysis
f_error = open('error_analysis.txt', 'w')

# Loop over methods and dt
for method, dts in cases.items():
    for dt in dts:
        # Read data from files
        U = []
        filename = f'./simulation-files/last-{method}-{dt}.txt'
        with open(filename, 'r') as f:
            for line in f:
                U.append(float(line))
        
        # Compute errors
        f_error.write(f'\nError between FE-0.005 and {method}-{dt}:\n')
        
        # Mean absolute error
        mae = np.mean(np.abs(np.array(U) - np.array(ref)))
        f_error.write(f'Mean absolute error: {mae:.4f}\n')
        
        # Mean square error
        mse = np.mean((np.array(U) - np.array(ref))**2)
        f_error.write(f'Mean square error: {mse:.4f}\n')
        
        # Relative error
        relative_error = np.linalg.norm(np.array(U) - np.array(ref)) / np.linalg.norm(np.array(ref))
        f_error.write(f'Relative error: {relative_error:.4f} ({100*relative_error:.4f} %)\n')