"""
@pgeorgantopoulos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LN
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from HighLevelFeatures import HighLevelFeatures as HLF

class HighLevelFeaturesExtended(HLF):
    """
    Wrapper class to HighLevelFeatures to support Point-Cloud visualization

    """
    def __init__(self, particle_type, binning_xml=None):
        super().__init__(particle_type, binning_xml)


    # def Calo2UniGrid(self, calo_data):
    #     """ Converts CaloChallenge data (2D array) to Universal Grid Representation (3D array) by incorporating the angular and radial binning information from the .xml 
        
    #         calo_data: 2D array of shape (num_events, num_cells)
    #         returns: 4D array of shape (num_events, num_rad_splits, num_ang_splits, num_layers)
    #     """
    #     num_events = calo_data.shape[0]
        
    #     num_layers = len(self.relevantLayers)
    #     uni_grid = []
    #     layer_boundaries = np.unique(self.bin_edges)
    #     for i_idx in range(len(calo_data.shape[0])):
        
                    
            
    #     return np.array(uni_grid)  # shape (num_layers, num_events, num_a_bins, num_r_bins)


    def DrawPC(self, data, filename, title):
        """ Draws the shower in all layers """
        if self.particle == 'electron':
            figsize = (10, 20)
        else:
            figsize = (len(self.relevantLayers)*2, 3)
        fig = plt.figure(figsize=figsize, dpi=200)
        # to smoothen the angular bins (must be multiple of self.num_alpha):
        num_splits = len(self.r_edges[0]) + 1
        layer_boundaries = np.unique(self.bin_edges)
        max_r = np.array(self.r_edges[0]).max()
        # max_r = 0
        # for radii in self.r_edges:
        #     if radii[-1] > max_r:
        #         max_r = radii[-1]
        vmax = data.max()
        for idx, layer in enumerate(self.relevantLayers):
            radii = np.array(self.r_edges[idx])
            if self.particle != 'electron':
                radii[1:] = np.log(radii[1:])
            data_reshaped = data[layer_boundaries[idx]:layer_boundaries[idx+1]].reshape(int(self.num_alpha[idx]), -1)

            
            r_bins = data_reshaped.shape[1]
            a_bins = data_reshaped.shape[0]
            theta_edges = 2. * np.pi * np.arange(a_bins + 1) / a_bins
            rad_centroids = []
            theta_centroids = []
            values = []
            for a in range(a_bins):
                theta_c = 0.5*(theta_edges[a] + theta_edges[a+1])
                for r in range(r_bins):
                    # polar area-weighted centroid for a sector
                    r_min = radii[r]
                    r_max = radii[r+1]
                    r_c = (2/3) * (r_max**3 - r_min**3) / (r_max**2 - r_min**2)
                    rad_centroids.append(r_c)
                    theta_centroids.append(theta_c)
                    values.append(data_reshaped[a, r])
            rad_centroids = np.array(rad_centroids)
            theta_centroids = np.array(theta_centroids)
            values = np.array(values)

            if self.particle == 'electron':
                ax = plt.subplot(9, 5, idx+1, polar=True)
            else:
                ax = plt.subplot(1, len(self.r_edges), idx+1, polar=True)
            plt.subplots_adjust(
                left=0.05,    # space on the left of figure
                right=0.95,   # space on the right
                top=0.9,      # space at the top
                bottom=0.1,   # space at the bottom
                wspace=0.5,   # horizontal space between subplots
                hspace=0.5    # vertical space (for multi-row layout)
            )
            ax.grid(False)
            sc = ax.scatter(theta_centroids, rad_centroids, c=values + 1e-16, s=0.1, norm=LN(vmin=1e-2, vmax=vmax))
            ax.axes.get_xaxis().set_visible(1)
            ax.set_xticks(np.deg2rad([0, 90,180,270]))
            ax.set_xticklabels([f"{a}°" for a in [0, 90,180,270]])
            [label.set_fontsize(5) for label in ax.axes.get_xaxis().get_ticklabels()]
            ax.axes.get_yaxis().set_visible(0)
            if self.particle == 'electron':
                ax.set_rmax(max_r)
            else:
                ax.set_rmax(np.log(max_r))
            ax.set_title('Layer '+str(layer)+'\n ang. splts:'+str(self.num_alpha[idx])+'\n rad. splts:'+str(radii.shape[0]-1))
        if self.particle == 'electron':
            axins = inset_axes(fig.get_axes()[-3], width="500%",
                                height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                                bbox_transform=fig.get_axes()[-3].transAxes,
                                borderpad=0)
        else:
            wdth = str(len(self.r_edges)*100)+'%'
            axins = inset_axes(fig.get_axes()[len(self.r_edges)//2], width=wdth,
                                height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                                bbox_transform=fig.get_axes()[len(self.r_edges)//2].transAxes,
                                borderpad=0)
        cbar = plt.colorbar(sc, cax=axins, fraction=0.2, orientation="horizontal")
        cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        if title is not None:
            plt.gcf().suptitle(title)
        if filename is not None:
            plt.savefig(filename, facecolor='white')
        else:
            plt.show()
        plt.close()


    def DrawSingleShowerNum(self,data,filename,title):
        """ Draws the shower in all layers with numbmering on the cells """

        # to smoothen the angular bins (must be multiple of self.num_alpha):
        num_splits = 400
        layer_boundaries = np.unique(self.bin_edges)
        max_r = 0
        for radii in self.r_edges:
            if radii[-1] > max_r:
                max_r = radii[-1]
        vmax = data.max()
        for idx, layer in enumerate(self.relevantLayers):
            if self.particle == 'electron':
                figsize = (10, 20)
            else:
                figsize = (len(self.relevantLayers)*2, 3)
            fig = plt.figure(figsize=figsize, dpi=300)
            radii = np.array(self.r_edges[idx])
            if self.particle != 'electron':
                radii[1:] = np.log(radii[1:])
            theta, rad = np.meshgrid(2.*np.pi*np.arange(num_splits+1)/ num_splits, radii)
            pts_per_angular_bin = int(num_splits / self.num_alpha[idx])
            data_reshaped = data[layer_boundaries[idx]:layer_boundaries[idx+1]].reshape(
                int(self.num_alpha[idx]), -1)
            data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)
            
            r_bins = data_reshaped.shape[1]
            a_bins = data_reshaped.shape[0]
                        
            theta_edges = 2. * np.pi * np.arange(a_bins+1) / (a_bins)
            rad_centroids = []
            theta_centroids = []
            for a in range(a_bins):
                theta_c = 0.5*(theta_edges[a] + theta_edges[a+1])
                for r in range(r_bins):
                    #  fpolar area-weighted centroidor a sector
                    r_min = radii[r]
                    r_max = radii[r+1]
                    r_c = (2/3) * (r_max**3 - r_min**3) / (r_max**2 - r_min**2)
                    rad_centroids.append(r_c)
                    theta_centroids.append(theta_c)
            rad_centroids = np.array(rad_centroids)
            theta_centroids = np.array(theta_centroids)

            if self.particle == 'electron':
                ax = plt.subplot(9, 5, idx+1, polar=True)
            else:
                ax = plt.subplot(1, 1, 1, polar=True)
            
            ax.grid(False)
            pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=1e-2, vmax=vmax))
            for i,(t_c,r_c) in enumerate(zip(theta_centroids, rad_centroids)):
                ax.text(
                    t_c,
                    r_c,
                    f"{i+len(self.r_edges[idx-1])*self.num_alpha[idx-1] - 1 if idx > 0 else i }",
                    ha="center",
                    va="center",
                    fontsize=3,
                    color="white"
                )
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if self.particle == 'electron':
                ax.set_rmax(max_r)
            else:
                ax.set_rmax(np.log(max_r))
            ax.set_title('Layer '+str(layer)+'\n ang. splts:'+str(self.num_alpha[idx])+'\n rad. splts:'+str(radii.shape[0]-1))
        # if self.particle == 'electron':
        #     axins = inset_axes(fig.get_axes()[-3], width="500%",
        #                        height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
        #                        bbox_transform=fig.get_axes()[-3].transAxes,
        #                        borderpad=0)
        # else:
        #     wdth = str(len(self.r_edges)*100)+'%'
        #     axins = inset_axes(fig.get_axes()[len(self.r_edges)//2], width=wdth,
        #                        height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
        #                        bbox_transform=fig.get_axes()[len(self.r_edges)//2].transAxes,
        #                        borderpad=0)
        # cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
        # cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        if title is not None:
            plt.gcf().suptitle(title)
        if filename is not None:
            plt.savefig(filename, facecolor='white')
        else:
            plt.show()
        plt.close()
