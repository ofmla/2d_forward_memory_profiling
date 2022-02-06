import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.ticker as plticker
import matplotlib as mpl

import numpy as np
import h5py
import json

if __name__ == '__main__':

    path = './marmousi2/parameters_hdf5/'
    l = [path+'vp.h5', path+'vp_start.h5', './final_result_LB.h5']

    with h5py.File(l[1], 'r') as f:
        metadata = json.loads(f['metadata'][()])
        shape = metadata['shape']
        im2 = np.empty(shape)
        im2[:] = f['vp_start'][()]

    im1 = np.empty(shape)
    im3 = np.empty(shape)

    images = [im1, im2, im3]

    for file, par in zip(l[0::2], images[0::2]):
        with h5py.File(file, 'r') as f:
            par[:] = f['vp'][()]

    label = ['a', 'b', 'c']
    label_format = '{:,.1f}'
    ticks_ylabels = [0.0, 1.0, 2.0, 3.]
    ticks_yloc = [0.0, 25.0, 50.0, 75.0]
    ticks_xlabels = [0.0, 4.0, 8.0, 12., 16.]
    ticks_xloc = [0.0, 100.0, 200.0, 300.0, 400.0]

    fig = plt.figure(figsize=(15, 20), constrained_layout=False)
    mpl.rcParams['font.size'] = 18

    spec = fig.add_gridspec(ncols=3, nrows=3)
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[1, 0:3])
    ax3 = fig.add_subplot(spec[2, 0:3])

    axes = [ax1, ax2, ax3]
    for count, ax in enumerate(axes):
        ax.text(5, 5, label[count],
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='1', edgecolor='none', pad=2))
        im = ax.imshow(images[count].T, vmax=4.688, vmin=1.377,
                       cmap=plt.cm.jet, aspect='auto')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="3%", pad="1%")
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params('both', length=2, width=0.5, which='major', labelsize=14)
        ax.yaxis.set_major_locator(plticker.FixedLocator(ticks_yloc))
        ax.xaxis.set_major_locator(plticker.FixedLocator(ticks_xloc))
        ax.set_yticklabels([label_format.format(x) for x in ticks_ylabels])
        ax.set_xticklabels([label_format.format(x) for x in ticks_xlabels])
        ax.tick_params('both', length=2, width=0.5, which='major')
        ax.set(xlabel='Distância (km)')
        ax.set(ylabel='Profundidade (km)')
        ax.label_outer()

    fig.savefig('fwi_marmosui2.pdf', bbox_inches="tight")

