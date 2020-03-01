import os
import numpy as np
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa


def save_scatter(samps, savedir, ename=''):
    nsets = len(samps)
    nplots = 3
    nfigs = 3
    for fcnt in range(nfigs):
        fig = plt.figure(figsize=(8, 4*nplots))
        pcnt = 0
        for ii in range(nplots):
            # Get sample and axis lengths (for equal axis)
            i = np.random.randint(nsets)
            x = samps[i]
            max_range = np.array(
                [x[:, 0].max()-x[:, 0].min(), x[:, 1].max()-x[:, 1].min(),
                    x[:, 2].max()-x[:, 2].min()]
            ).max() / 2.0
            mid_x = (x[:, 0].max()+x[:, 0].min()) * 0.5
            mid_y = (x[:, 1].max()+x[:, 1].min()) * 0.5
            mid_z = (x[:, 2].max()+x[:, 2].min()) * 0.5
            # Angle 1
            pcnt += 1
            ax = fig.add_subplot(nplots*100+30+pcnt, projection='3d')
            ax.scatter(
                x[:, 0], x[:, 1], x[:, 2],
                c='red', edgecolors='k', alpha=0.4)
            ax.view_init(30, 45)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            # Angle 2
            pcnt += 1
            ax = fig.add_subplot(nplots*100+30+pcnt, projection='3d')
            ax.scatter(
                x[:, 0], x[:, 1], x[:, 2],
                c='red', edgecolors='k', alpha=0.4)
            ax.view_init(30, 225)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_title('Samples')
            # Angle 3
            pcnt += 1
            ax = fig.add_subplot(nplots*100+30+pcnt, projection='3d')
            ax.scatter(
                x[:, 0], x[:, 1], x[:, 2],
                c='red', edgecolors='k', alpha=0.4)
            ax.view_init(azim=0, elev=90)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Save figure
        fig_path = os.path.join(
            savedir, 'samp_{}_{}.png'.format(fcnt, ename))
        fig.savefig(fig_path)
