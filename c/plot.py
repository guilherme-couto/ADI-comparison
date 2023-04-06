import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import imageio.v2

if len(sys.argv) != 3:
    print('Usage: python3 plot.py <method> <dt>')
    sys.exit(1)

method = sys.argv[1]
dt = sys.argv[2]

# Read data from files

t = []
timesfile = f'./simulation-files/sim-times-{method}-{dt}.txt'
with open(timesfile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        t.append(float(line))

totalframes = len(t)

filename = f'./simulation-files/fhn-{method}-{dt}.txt'
U = np.zeros((totalframes, 100, 100))
with open(filename, 'r') as f:
    for n in range(totalframes):
        for i in range(len(U[0])):
            for j in range(len(U[0])):
                U[n][i][j] = float(f.readline())


# Make plots
framerate = int(totalframes / 100)
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
with imageio.v2.get_writer(f'./gif/gif-{method}-{dt}.gif', mode='I') as writer:
    for plot in plots:
        image = imageio.v2.imread(plot)
        writer.append_data(image)
        
# Remove files
for png in set(plots):
    os.remove(png)

    
    