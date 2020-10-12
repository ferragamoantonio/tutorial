import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import biweight
from astropy.visualization import hist
from astropy.stats import histogram
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.spatial.transform import Rotation as R
import scipy.odr as o
import bces.bces
import csv
import statsmodels.api as sm
import ai as ai
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy.table import Table


def corr_sigma(sigma_s, sigma_g, sigma_b, ngal):
    std = sigma_s*(1.+1/4./(ngal-1.)-0.0037)
    gap = sigma_g*(1.- 0.008)
    bwt = sigma_b*(1.+((0.72/(ngal-1.))**1.28)-0.0225)
    return std, gap, bwt


#def log_likelihood_no_is(theta, x, y, sig_x, sig_y):
def log_likelihood_no_is(theta, x, y, sig_y):
    a, b = theta
    model  = a * x + b
    sigma2 = sig_y**2
    return -0.5 * np.sum( (y - model)**2 / sigma2  )


def int_scat_est(par, x, y, sig_y):
    #[4] Quick estimate of intrinsic scatter, based on Tremaine et al. (2020), ApJ 574, 740.
    #    Computing sig_int to make chi2_dof = 1
    #theta    = np.array( [np.mean(flat_samples2[:,0]), np.mean(flat_samples2[:,1]) ] )
    theta = np.array([par[0], np.log10(par[1])])
    chi2_dof = -2.0 * log_likelihood_no_is(theta, x, y, sig_y) / (len(x)-2.)
    sig_int_est2 = 0
    if chi2_dof > 1:
        s0min = 0
        #s0max = np.sqrt(chi2_dof)*np.max( np.sqrt(sig_y**2 + (theta[0]*sig_x)**2))
        s0max = np.sqrt(chi2_dof)*np.max( np.sqrt(sig_y**2))
        ns0   = 10000
        step0 = (s0max-s0min)/ns0
        chi2_v = np.zeros(ns0)
        s0_v   = np.arange(s0min,s0max,step0)
        for j in np.arange(ns0):
            s0        = s0_v[j]
            chi2_v[j] = -2.0 * log_likelihood_no_is(theta, x, y, np.sqrt(sig_y**2 + s0**2)) / (len(x)-2.)
        dist = np.abs( chi2_v - 1.0 )
        z = s0_v[ np.where(dist == np.amin(dist))]
        sig_int_est2 = z[0]
    #print('sigma_int = {0:.3f}'.format(sig_int_est2))
    return sig_int_est2

def hist_normalised(y,ax=None,**kwargs):
    #hh, bi =  histogram(np.array(p1.members_velocities200[iproj]), bins='scott', density=True)#histtype='stepfilled',alpha=0.2)#, density=True)
    hh, bi =  histogram(y, bins='scott', density=True)
    bin_widths = bi[1:]-bi[:-1]
    bin_centers = 0.5*(bi[:-1]+bi[1:])
    hist1b = hh/np.max(hh)

    bin_edge = np.append(bin_centers[0]-bin_widths[0],bin_centers)
    bin_edge = np.append(bin_edge, bin_centers[-1]+bin_widths[-1])
    h1_edge = np.append([0],hist1b)
    h1_edge = np.append(h1_edge, [0])
    ax.step(bin_edge, h1_edge, where='mid', color=kwargs.get("color", 0))#'C'+str(iproj))
    ax.bar(bin_centers, hist1b, width = bin_widths, align = 'center', color=kwargs.get("color", 0), alpha = 0.5)
    #ax.bar(bin_centers, hist1b, width = bin_widths, align = 'center', color='C'+str(iproj), alpha = 0.5)
    return hist1b, bin_centers

def ho19(m):
    A = 1087.
    b = 0.366
    return A * (m)**b # giving mass in h^-1 Msun   (SUB)

def munari13(m, tracer='sub'):
    if tracer == 'gal':
        A = 1177.
        b = 0.364
    elif tracer == 'sub':
        A = 1199.
        b = 0.365
    return A * (m)**b # giving mass in h^-1 Msun   (SUB)

def saro13(m):
    #return 939 * (m)**(1/2.91) *(67.7/70)**0.33
    return 939 * (m/0.677)**(1/2.91) #*(67.7/70)**0.33

def lin_func(x, a, b):
    return a*(x) + b

def pow_law_func(x, a, b):
    return b * (x)**a
    #return b * (0.677*x)**a

def linear_fit(tt,xx, yy, eyy, method = 'ls'):
    #xx = 0.67*xx
    log_x = np.log10(xx)
    log_y = np.log10(yy)
    log_e_y = eyy/yy/np.log(10.)

    if method == 'ls':
        popt, pcov = curve_fit(lin_func, log_x, log_y, sigma=log_e_y)
        a, b = popt
        ea, eb = np.sqrt(np.diag(pcov))

    elif method == 'bces':
        log_e_x = eyy*1.e-20/np.log(10.)/yy
        cov = np.zeros_like(log_x)
        a_bces,b_bces,aerr_bces,berr_bces,covab = bces.bces.bces(log_x, log_e_x, log_y, log_e_y,cov)
        a = a_bces[3]
        ea = aerr_bces[3]
        b = b_bces[3]
        eb = berr_bces[3]
        #b = 10.**b_bces[3]
        #e_b = berr_bces[3] * 10.**b_bces[3] * np.log(10)

    elif method == 'siegel_h':
        a,b = stats.siegelslopes(log_y, log_x)
        eb, ea = 0, 0
    elif method == 'siegel_s':
        a,b = stats.siegelslopes(log_y, log_x, method = 'separate')
        eb, ea = 0, 0

    elif method == 'theil_sen':
        a,b, am, ap = stats.theilslopes(log_y, log_x, 0.68)
        eb, ea = a-am, 0

    elif method == 'rlm':
        log_X = sm.add_constant(log_x)
        resrlm = sm.RLM(log_y, log_X).fit()
        b, a = resrlm.params
        eb, ea = resrlm.bse

    #a,b = popt
    #ea ,eb = np.sqrt(np.diag(pcov))
    par = [a, 10**b]
    per = [ea, 10.**b * np.log(10) * eb]
    fit = pow_law_func(tt, par[0], par[1])
    return par, per, fit

def gapper(v):
    """ Returns the gapper velocity dispersion of a cluster (Sigma_G)

    v is an array of galaxy velocity values.
    """
    v=np.array(v)
    vs=v[np.argsort(v)]
    n = len(vs)
    w = np.arange(1, n) * np.arange(n-1, 0, -1)
    g = np.diff(vs)
    sigG = (np.sqrt(np.pi))/(n*(n-1)) * np.dot(w, g)
    return sigG

def errors_estim(v):
    ss = []
    sg = []
    sb = []
    for ib in range(100):
        np.random.seed(1986*ib)
        vv = np.random.choice(v, len(v), replace=True)
        ss = np.append(ss, np.std(vv, ddof=1))
        sg = np.append(sg, gapper(vv))
        sb = np.append(sb, biweight.biweight_scale(vv))#, c=9.0, M=biweight.biweight_location(vv, M=np.mean(vv))))
        #if len(v) > 1:
        #    sb = np.append(sb, aux_sb*np.sqrt(len(vv)/(len(vv)-1)))
        #else: sb = np.append(sb, aux_sb)
        #sb = np.append(sb, biweight.biweight_scale(vv, c=9.0, M=biweight.biweight_location(vv, M=np.mean(vv))))#*np.sqrt(len(vv)/(len(vv)-1)))

    e_ss = np.std(ss)
    e_sg = np.std(sg)
    e_sb = np.std(sb)
    return e_ss, e_sg, e_sb

class Cluster:
    def __init__(self, path, name, tracer='sub'):
        #self.mass200 = mass200
        self.name = name
        self.data = np.load(path+name)
        self.header = self.data[0]
        self.id_cluster = self.data['id_cluster'][0]
        self.R200 = self.data['vir_radius'][0]*1.e-3
        self.M200 = self.data['mass'][0]
        self.n_gas = self.data['n_gas'][0]
        self.M_gas = self.data['M_gas'][0]
        self.n_stars = self.data['n_stars'][0]
        self.M_stars = self.data['M_stars'][0]
        self.n_tot = self.data['n_tot'][0]
        self.n_DM = self.n_tot - self.n_gas - self.n_stars
        self.cluster_center = [self.data['x'][0]*1.e-3, self.data['y'][0]*1.e-3, self.data['z'][0]*1.e-3]
        self.red = 0 #self.data['redshift'][0]

        if tracer == 'gal': self.is_gal = np.where(self.data['mag_r'] != 0)[0]
        elif tracer == 'sub': self.is_gal = np.indices((len(self.data['mag_r']),))[0]
        self.gmag = self.data['mag_g'][self.is_gal]
        self.rmag = self.data['mag_r'][self.is_gal]
        self.imag = self.data['mag_i'][self.is_gal]


        self.id_gal = self.data['id_cluster'][self.is_gal]
        self.members_id = self.data['id_cluster'][self.is_gal]

        self.tot_members = len(self.id_gal)

        self.n_stars_mem = self.data['n_stars'][self.is_gal]
        self.M_stars_mem = self.data['M_stars'][self.is_gal]
        self.M200_mem = self.data['mass'][self.is_gal]


        self.members_coord_x = self.cluster_center[0]-self.data['x'][self.is_gal]*1.e-3
        self.members_coord_y = self.cluster_center[1]-self.data['y'][self.is_gal]*1.e-3
        self.members_coord_z = self.cluster_center[2]-self.data['z'][self.is_gal]*1.e-3
        self.members_coord = np.dstack((self.members_coord_x,self.members_coord_y,self.members_coord_z)).reshape(self.tot_members,3)


        self.members_vx = self.data['vx'][self.is_gal]
        self.members_vy = self.data['vy'][self.is_gal]
        self.members_vz = self.data['vz'][self.is_gal]
        #self.members_velocities = np.dstack((self.members_vx,self.members_vy,self.members_vz)).reshape(self.tot_members,3)#.reshape(self.tot_members,3)
        self.members_velocities = np.dstack((self.members_vx,self.members_vy,self.members_vz)).reshape(self.tot_members,3)
        #self.members_velocities = np.vstack((self.members_vx, self.members_vy, self.members_vz))

        self.dist3d = self.cyl_sphe_coord(self.members_coord_x, self.members_coord_y, self.members_coord_z)[1]

        self.tot_members200 = len(self.id_gal[self.dist3d <= self.R200])
        #self.members_velocities200 = np.vstack((self.members_vx[self.dist3d <= self.R200], self.members_vy[self.dist3d <= self.R200], self.members_vz[self.dist3d <= self.R200]))
        self.members_velocities200 = np.dstack((self.members_vx[self.dist3d <= self.R200], self.members_vy[self.dist3d <= self.R200], self.members_vz[self.dist3d <= self.R200])).reshape(self.tot_members200,3)
        self.members_velocities200_2 = np.vstack((self.members_vx[self.dist3d <= self.R200], self.members_vy[self.dist3d <= self.R200], self.members_vz[self.dist3d <= self.R200]))

        self.z_proj_coords = np.append(self.cyl_sphe_coord(self.members_coord_x, self.members_coord_y, self.members_coord_z)[:2],self.members_vz).reshape(3,self.tot_members)
        self.y_proj_coords = np.append(self.cyl_sphe_coord(self.members_coord_x, self.members_coord_z, self.members_coord_y)[:2],self.members_vy).reshape(3,self.tot_members)
        self.x_proj_coords = np.append(self.cyl_sphe_coord(self.members_coord_y, self.members_coord_z, self.members_coord_x)[:2],self.members_vx).reshape(3,self.tot_members)

        self.e_sigma_los_s, self.e_sigma_los_g, self.e_sigma_los_b = self.err_sigma_rot(self.members_coord, self.members_velocities, self.R200)

    def err_sigma_rot(self, coord, velo, r):
        rott = [[0,0,0],[-90,-90,0],[-90,-180,-90],[-90,-135,0],[-90,-45,0],[-45,0,0],[45,0,0],[45,-45,0],[45,45,0],[-45,-45,0],[-45,45,0]]
        los_sig_s = []
        los_sig_g = []
        los_sig_b = []
        for irot in range(len(rott)):
            los_proj = self.los_projection(coord, velo, proj=irot)
            aux_sigma = self.los_sigmas_sph(los_proj, r)
            los_sig_s = np.append(los_sig_s, aux_sigma[0][0])
            los_sig_g = np.append(los_sig_g, aux_sigma[1][0])
            los_sig_b = np.append(los_sig_b, aux_sigma[2][0])

        return np.std(los_sig_s), np.std(los_sig_g), np.std(los_sig_b)

    def cluster_rotation(self,coord, velo, proj=0):
        rott = [[0,0,0],[-90,-90,0],[-90,-180,-90],[-90,-135,0],[-90,-45,0],[-45,0,0],[45,0,0],[45,-45,0],[45,45,0],[-45,-45,0],[-45,45,0]]
        r = R.from_euler('xyz', rott[proj], degrees=True)

        self.rot_coord = r.apply(coord)
        self.rot_velo = r.apply(velo)
        return self.rot_coord, self.rot_velo

    def los_projection(self,coord, velo, proj=0):
        r_coord, r_velo = self.cluster_rotation(coord, velo, proj=proj)

        rc, rs, theta, phi = self.cyl_sphe_coord(r_coord[:,0],r_coord[:,1],r_coord[:,2])
        self.los_proj = np.dstack((rc, rs, r_coord[:,0],r_coord[:,1],r_velo[:,2])).reshape(len(r_coord[:,0]),5)
        return self.los_proj

    def los_sigmas_sph(self,proj_info, r):
        v = proj_info[:,4][proj_info[:,1] <= r]
        self.sigma_los_sph_s = np.std(v, ddof=1)
        self.sigma_los_sph_g = gapper(v)
        self.sigma_los_sph_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_los_sph_b = aux_bwt*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_los_sph_b = aux_bwt
        #self.sigma_los_sph_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))*np.sqrt(len(v)/(len(v)-1))

        #ERRORS
        self.e_sigma_los_sph_s, self.e_sigma_los_sph_g, self.e_sigma_los_sph_b = errors_estim(v)
        return (self.sigma_los_sph_s, self.e_sigma_los_sph_s) , (self.sigma_los_sph_g, self.sigma_los_sph_g), (self.sigma_los_sph_b, self.sigma_los_sph_b)

    def los_sigmas_cyl(self,proj_info, r):
        v = proj_info[:,4][proj_info[:,0] <= r]

        self.sigma_los_cyl_s = np.std(v, ddof=1)
        self.sigma_los_cyl_g = gapper(v)
        self.sigma_los_cyl_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_los_cyl_b = aux_bwt*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_los_cyl_b = aux_bwt

        #self.sigma_los_cyl_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))*np.sqrt(len(v)/(len(v)-1))

        #ERRORS
        self.e_sigma_los_cyl_s, self.e_sigma_los_cyl_g, self.e_sigma_los_cyl_b = errors_estim(v)
        return (self.sigma_los_cyl_s, self.e_sigma_los_cyl_s) , (self.sigma_los_cyl_g, self.sigma_los_cyl_g), (self.sigma_los_cyl_b, self.sigma_los_cyl_b)

    def sigmas200(self,vv):
        #v = vv[0]
        v = vv[:,0]
        self.sigma_x_s = np.std(v, ddof=1)
        self.sigma_x_g = gapper(v)
        self.sigma_x_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_x_b = aux_bwt_x*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_x_b = aux_bwt_x

        #ERRORS
        e_ssx, e_sgx, e_sbx = errors_estim(v)

        #v = vv[1]
        v = vv[:,1]
        self.sigma_y_s = np.std(v, ddof=1)
        self.sigma_y_g = gapper(v)
        self.sigma_y_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))*np.sqrt(len(v)/(len(v)-1))
        #aux_bwt_y = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_y_b = aux_bwt_y*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_y_b = aux_bwt_y

        #ERRORS
        e_ssy, e_sgy, e_sby = errors_estim(v)

        #v = vv[2]
        v = vv[:,2]
        self.sigma_z_s = np.std(v, ddof=1)
        self.sigma_z_g = gapper(v)
        self.sigma_z_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_z_b = aux_bwt_z*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_z_b = aux_bwt_z

        #ERRORS
        e_ssz, e_sgz, e_sbz = errors_estim(v)

        self.sigma3d_s = np.sqrt(self.sigma_x_s**2+self.sigma_y_s**2+self.sigma_z_s**2)
        self.e_sigma3d_s = np.sqrt((self.sigma_x_s*e_ssx)**2+(self.sigma_y_s*e_ssy)**2+(self.sigma_z_s*e_ssz)**2)/self.sigma3d_s

        self.sigma1d_s = self.sigma3d_s/np.sqrt(3.)
        self.e_sigma1d_s = self.e_sigma3d_s/np.sqrt(3.)

        self.sigma3d_g = np.sqrt(self.sigma_x_g**2+self.sigma_y_g**2+self.sigma_z_g**2)
        self.e_sigma3d_g = np.sqrt((self.sigma_x_g*e_sgx)**2+(self.sigma_y_g*e_sgy)**2+(self.sigma_z_g*e_sgz)**2)/self.sigma3d_g

        self.sigma1d_g = self.sigma3d_g/np.sqrt(3.)
        self.e_sigma1d_g = self.e_sigma3d_g/np.sqrt(3.)

        self.sigma3d_b = np.sqrt(self.sigma_x_b**2+self.sigma_y_b**2+self.sigma_z_b**2)
        self.e_sigma3d_b = np.sqrt((self.sigma_x_b*e_sbx)**2+(self.sigma_y_b*e_sby)**2+(self.sigma_z_b*e_sbz)**2)/self.sigma3d_b

        self.sigma1d_b = self.sigma3d_b/np.sqrt(3.)
        self.e_sigma1d_b = self.e_sigma3d_b/np.sqrt(3.)

        return (self.sigma3d_s, self.e_sigma3d_s) , (self.sigma3d_g, self.e_sigma3d_g), (self.sigma3d_b, self.e_sigma3d_b), (self.sigma1d_s, self.e_sigma1d_s) , (self.sigma1d_g, self.e_sigma1d_g), (self.sigma1d_b, self.e_sigma1d_b)


    def cyl_sphe_coord(self, x,y,z):
        xy = x**2 + y**2
        self.rc = np.sqrt(xy)
        self.rs = np.sqrt(xy + z**2)
        self.phi = np.arctan2(y, x)  # theta (nel piano x,y)
        self.theta = np.arctan2(z, np.sqrt(xy)) # for elevation angle defined from Z-axis down
        #self.phi = np.arctan2(z, np.sqrt(xy + z**2))  # for elevation angle defined from XY-plane up
        #ptsnew[:,5] = np.arctan2(z, rc) # for elevation angle defined from XY-plane up
        return self.rc, self.rs, self.theta, self.phi

    def velocity_disp_proj(self, pc, radius):
        v = pc[2][pc[0]<=radius]
        self.sigma_s = np.std(v, ddof=1)
        self.sigma_g = gapper(v)
        self.sigma_b = biweight.biweight_scale(v)#, c=9.0, M=biweight.biweight_location(v, M=np.mean(v)))
        #if len(v) > 1:
        #    self.sigma_b = aux_bwt*np.sqrt(len(v)/(len(v)-1))
        #else: self.sigma_b = aux_bwt

        return self.sigma_s, self.sigma_g, self.sigma_b
