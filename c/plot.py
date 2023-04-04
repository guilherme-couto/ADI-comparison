import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import imageio.v2

if len(sys.argv) != 5:
    print('Usage: python3 plot.py <method> <dt> <total_frames> <frame_rate>')
    sys.exit(1)

method = sys.argv[1]
dt = sys.argv[2]
totalframes = int(sys.argv[3])
framerate = int(sys.argv[4])

U = np.zeros((totalframes, 100, 100))
t = np.zeros(totalframes)

filename = f'fhn-{method}-{dt}.txt'
timesfile = f'sim-times-{method}-{dt}.txt'

# Read data from files
with open(filename, 'r') as f:
    for n in range(totalframes):
        for i in range(len(U[0])):
            for j in range(len(U[0])):
                U[n][i][j] = float(f.readline())

with open(timesfile, 'r') as f:
    for n in range(totalframes):
        t[n] = float(f.readline())

# Make plots
plots = []
for n in range(len(U)):
    if n % framerate == 0:
        plotname = 'plot-' + str(n) + '.png'
        plots.append(plotname)
        
        plt.imshow(U[n], cmap='plasma', vmin=0, vmax=100)
        plt.colorbar(label='V (mV)')
        plt.title(f'Monodomain FHN {method} dt = {dt} t = {t[n]:.2f}')
        plt.xticks([])
        plt.yticks([])
        
        plt.savefig(plotname)
        plt.close()

# Build gif
with imageio.v2.get_writer(f'gif-{method}-{dt}.gif', mode='I') as writer:
    for plot in plots:
        image = imageio.v2.imread(plot)
        writer.append_data(image)
        
# Remove files
for png in set(plots):
    os.remove(png)

    
    