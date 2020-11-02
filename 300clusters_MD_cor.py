#300clusters_MD_cor.py


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
from cluster_ob import *

plt.close('all')
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.autolayout': True})

ired = 0
red_dirs = ['z0_splitted', 'z04_splitted', 'z05_splitted', 'z06_splitted', 'z1_splitted']
redshifts = [0.0, 0.4, 0.5, 0.6, 1.0]
red_dir = red_dirs[ired]
z_red = redshifts[ired]

file_path = "/Users/antonioferragamo/Desktop/300_Clusters/sim_cat/"+red_dir+"/"
with open(file_path+'clusters_list.txt', 'r') as clu_file:
    clu_reader = csv.reader(clu_file)
    for row in clu_reader:
        clusters_list = row

with open(file_path+'clusters_list_central.txt', 'r') as clu_file:
    clu_reader = csv.reader(clu_file)
    for row in clu_reader:
        clusters_list_central = row

saveplot = 0
#tracers = 'sub'
tracers = 'gal'
#non_comulative = False
non_comulative = True

n_clusters = 325

"""
### PRENDERE INFO DEI CLUSTER CENTRALI ###

fig, ax = plt.subplots()

fcl_center = []
fcl_R200 = []
fcl_M200 = []
kurt_3d = []
skew_3d = []
for ic, shalo in enumerate(clusters_list_central):
    #if ic == 2 :
    file_name = shalo
    if '_1_z' in file_name:
        fcl = Cluster(file_path, file_name, tracer=tracers)
        hh, bi = hist_normalised(np.array(fcl.members_velocities200[:,0]), ax=ax, color='C'+str(0))
        hh, bi = hist_normalised(np.array(fcl.members_velocities200[:,1]), ax=ax, color='C'+str(1))
        hh, bi = hist_normalised(np.array(fcl.members_velocities200[:,2]), ax=ax, color='C'+str(2))

        vel_sph = fcl.cyl_sphe_coord(fcl.members_velocities200[:,0],fcl.members_velocities200[:,1],fcl.members_velocities200[:,2])[1]
        kurt_3d = np.append(kurt_3d,kurtosis(vel_sph))
        skew_3d = np.append(skew_3d,skew(vel_sph))
            #hh, bi = hist_normalised(np.array(fcl.members_velocities[:,0]), ax=ax, color='C'+str(2))
        hz = cosmo.H(z_red).value/100.
        fcl_center.append(fcl.cluster_center)
        fcl_R200 = np.append(fcl_R200, fcl.R200)
        fcl_M200 = np.append(fcl_M200, fcl.M200/cosmo.h*hz)
fcl_center = np.array(fcl_center)


First_Clusters = Table()
First_Clusters['coordinates'] = fcl_center
First_Clusters['R200'] = fcl_R200
First_Clusters['M200'] = fcl_M200
First_Clusters.write(file_path+'central_clusters_coord_rad_mass.fits', overwrite=True)
"""

### LEGGERE INFO DEI CLUSTER CENTRALI ###

fcl = Table.read(file_path+'output/central_clusters_coord_rad_mass.fits')


star_mass_lims = [10**6, 10**8, 10**8.5, 10**9, 10**9.5, 10**9.8, 10**10, 10**10.25, 10**10.5]
star_mass_strs = ['6', '8', '8.5', '9', '9.5', '9.8', '10', '10.25', '10.5']

h_mass_lims = np.arange(13.6, 15.4, 0.2)
halo_mass_lim_names = ['13_6', '13_8', '14_0', '14_2', '14_4', '14_6', '14_8', '15_0', '15_2']
halo_mass_lim_names2 = ['13.6', '13.8', '14.0', '14.2', '14.4', '14.6', '14.8', '15.0', '15.2']

for i_s_mass_lim in range(len(star_mass_strs)):
#for i_s_mass_lim in [8]:
    #if i_s_mass_lim >= 1: break
    #i_s_mass_lim = 8
    star_mass_lim = star_mass_lims[i_s_mass_lim]
    stell_mass_str = star_mass_strs[i_s_mass_lim]

    for i_h_mass_lim in range(len(h_mass_lims)):
    #for i_h_mass_lim in [0]:
        if i_h_mass_lim > 8: break
        #if i_h_mass_lim >= 1: break
        mass_lim_name = halo_mass_lim_names[i_h_mass_lim]
        mass_lim_name2 = halo_mass_lim_names2[i_h_mass_lim]

        ngal_lim = 3

        print('')
        print('stell_mass = 10**'+stell_mass_str)
        print('halo_mass =  10**'+mass_lim_name2)

        cl_names = []
        masses = []
        M200 = []
        R200 = []
        M_stars = []
        n_stars = []

        sigma200_s1 = []
        e_sigma200_s = []
        e_sigma200_stat_s = []
        sigma200_g1 = []
        e_sigma200_g = []
        e_sigma200_stat_g = []
        sigma200_b1 = []
        e_sigma200_b = []
        e_sigma200_stat_b = []
        nmem = []
        nmem_mass = []

        #STARTS LOOP OVER SIMULATED CLUSTERS
        kurt = []
        skewn = []
        pvalue_s = []
        pvalue_g = []
        pvalue_b = []
        kurt_3d = []
        skew_3d = []
        pv_3d_s = []
        pv_3d_g = []
        pv_3d_b = []

        coord_fcl = []
        df_fcl = []
        df_fcl_r200 = []

        BCG_vel = []
        mean_vel = []
        mean_vel_mass = []

        iicc = 0
        cluster_names = []
        for ic, shalo in enumerate(clusters_list):
            file_name = shalo
            #if '_1_z' not in file_name: continue
            p1 = Cluster(file_path, file_name, tracer=tracers)
            setattr(p1, 'redshift', redshifts[ired])
            hz = cosmo.H(p1.redshift).value/100.

            nmem_cl = p1.tot_members200
            nmem_cl_mass = len(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),0])
            #nmem_cl_mass = len(p1.M_stars_mem[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim)])

            if nmem_cl_mass < ngal_lim: continue

            ## NON COMULATIVE
            if non_comulative:
                non_com_str = '_non_com'
                if np.log10(p1.M200/cosmo.h*hz) < h_mass_lims[i_h_mass_lim] or np.log10(p1.M200/cosmo.h*hz) >= h_mass_lims[i_h_mass_lim]+0.2: continue
            else: non_com_str = ''

            M_stars = np.append(M_stars, p1.M_stars_mem)
            n_stars = np.append(n_stars, p1.n_stars_mem)
            cluster_names.append(shalo)


            for i_fcl in (range(len(fcl['coordinates']))):
                if i_fcl < 306: si_fcl = str(i_fcl+1)
                else: si_fcl = str(i_fcl+2)
                if 'NewMDCLUSTER_'+si_fcl+'_' in p1.name:
                    aux_coord_flc = p1.cluster_center-fcl['coordinates'][i_fcl]
                    coord_fcl.append(aux_coord_flc)
                    aux_df = np.sqrt(aux_coord_flc[0]**2+aux_coord_flc[1]**2+aux_coord_flc[2]**2)
                    df_fcl.append(aux_df)
                    df_fcl_r200.append(aux_df/p1.R200)

            BCG_vel.append(p1.members_velocities[0,:])
            #vel_cl = p1.members_velocities200[1:,:]
            #vel_cl_mass = p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),:][1:]
            mean_vel.append(np.mean(p1.members_velocities200[1:,:], axis=0))
            mean_vel_mass.append(np.mean(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),:][1:], axis=0))

            cl_names.append(p1.name)
            nmem = np.append(nmem,nmem_cl)
            nmem_mass = np.append(nmem_mass,nmem_cl_mass)
            M200 = np.append(M200, p1.M200/1.e15/cosmo.h*hz)
            masses.append(p1.M200/cosmo.h*hz)
            R200 = np.append(R200, p1.R200)

            """
            #RIMUOVERE UN TAB QUANDO SI VUOLE USARE
            Clusters_infos = Table()
            Clusters_infos['cl_name'] = names
            Clusters_infos['M200'] = M200
            Clusters_infos['R200'] = R200
            Clusters_infos['df_fcl'] = np.array(df_fcl)
            Clusters_infos['df_fcl_norm_r200'] = np.array(df_fcl_r200)
            Clusters_infos['BCG_vel'] = np.array(BCG_vel)
            Clusters_infos['mean_vel'] = np.array(mean_vel)
            Clusters_infos['mean_vel_mass'] = np.array(mean_vel_mass)
            Clusters_infos['nmem'] = np.array(nmem)
            Clusters_infos['nmem_mass'] = np.array(nmem_mass)
            Clusters_infos.write(file_path+'output/%s/clusters_info_vel_dist_%s_%s.fits'%(tracers,stell_mass_str,mass_lim_name2), overwrite=True)
            """

            aux_sigma = p1.sigmas200(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),:])

            """
            np.random.seed(1986)
            vvx = np.random.choice(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),0], 10, replace=False)
            vvy = np.random.choice(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),1], 10, replace=False)
            vvz = np.random.choice(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),2], 10, replace=False)
            vv = np.dstack((vvx,vvy,vvz)).reshape(len(vvx),3)
            aux_sigma = p1.sigmas200(vv)
            """

            sigma200_s1 = np.append(sigma200_s1, aux_sigma[3][0])
            sigma200_g1 = np.append(sigma200_g1, aux_sigma[4][0])
            sigma200_b1 = np.append(sigma200_b1, aux_sigma[5][0]*np.sqrt(nmem_cl_mass/(nmem_cl_mass-1)))


            e_sigma200_stat_s = np.append(e_sigma200_stat_s, aux_sigma[3][1])
            e_sigma200_stat_g = np.append(e_sigma200_stat_g, aux_sigma[4][1])
            e_sigma200_stat_b = np.append(e_sigma200_stat_b, aux_sigma[5][1])#*np.sqrt(nmem_cl_mass/(nmem_cl_mass-1)))

            e_sigma200_s = np.append(e_sigma200_s, p1.e_sigma_los_s)
            e_sigma200_g = np.append(e_sigma200_g, p1.e_sigma_los_g)
            e_sigma200_b = np.append(e_sigma200_b, p1.e_sigma_los_b)

            vel_sph = p1.cyl_sphe_coord(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),0],p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),1],p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),2])[1]
            kurt_3d = np.append(kurt_3d,kurtosis(vel_sph))
            skew_3d = np.append(skew_3d,skew(vel_sph))
            np.random.seed(12345678)  #fix random seed to get the same result
            n1 = 1000  # size of first sample
            rvs_s = stats.norm.rvs(size=n1, loc=0., scale=1)
            rvs_g = stats.norm.rvs(size=n1, loc=0., scale=1)
            rvs_b = stats.norm.rvs(size=n1, loc=0., scale=1)
            dv_s, pv_s = stats.ks_2samp((vel_sph-np.mean(vel_sph))/aux_sigma[3][0], rvs_s)
            dv_g, pv_g = stats.ks_2samp((vel_sph-np.mean(vel_sph))/aux_sigma[4][0], rvs_g)
            dv_b, pv_b = stats.ks_2samp((vel_sph-np.mean(vel_sph))/aux_sigma[5][0], rvs_b)
            pv_3d_s = np.append(pv_3d_s, pv_s)
            pv_3d_g = np.append(pv_3d_g, pv_g)
            pv_3d_b = np.append(pv_3d_b, pv_b)


            kur_proj = []
            skewn_proj = []
            pv_proj_s = []
            pv_proj_g = []
            pv_proj_b = []
            for iproj in range(3):
                kur_proj.append(kurtosis(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),iproj]))
                skewn_proj.append(skew(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),iproj]))

                np.random.seed(12345678)  #fix random seed to get the same result
                n1 = 1000  # size of first sample
                rvs_s = stats.norm.rvs(size=n1, loc=0., scale=1)
                rvs_g = stats.norm.rvs(size=n1, loc=0., scale=1)
                rvs_b = stats.norm.rvs(size=n1, loc=0., scale=1)
                dv_s, pv_s = stats.ks_2samp(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),iproj]/aux_sigma[3][0], rvs_s)
                dv_g, pv_g = stats.ks_2samp(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),iproj]/aux_sigma[4][0], rvs_g)
                dv_b, pv_b = stats.ks_2samp(p1.members_velocities[np.logical_and(p1.dist3d <= p1.R200, p1.M_stars_mem >= star_mass_lim),iproj]/aux_sigma[5][0], rvs_b)

                pv_proj_s = np.append(pv_proj_s, pv_s)
                pv_proj_g = np.append(pv_proj_g, pv_g)
                pv_proj_b = np.append(pv_proj_b, pv_b)


            kurt.append(kur_proj)
            skewn.append(skewn_proj)

            pvalue_s.append(pv_proj_s)
            pvalue_g.append(pv_proj_g)
            pvalue_b.append(pv_proj_b)

            iicc += 1
            p1.cyl_sphe_coord(p1.members_coord_x, p1.members_coord_y, p1.members_coord_z)

        kurt = np.array(kurt)
        skewn = np.array(skewn)

        lab = ['vx','vy','vz']

        notnan = np.where(sigma200_s1 != 0)
        mtest = np.log10(np.logspace(0.01,7, base=10))

        fit_names =['ls', 'bces', 'rlm', 'siegel_h', 'siegel_s', 'theil_sen']
        fit_colors = ['deepskyblue', 'red', 'green', 'limegreen', 'blue', 'mediumpurple']

        par_s = []
        per_s = []
        fit_s = []
        par_g = []
        per_g = []
        fit_g = []
        par_b = []
        per_b = []
        fit_b = []

        sigma200_s, sigma200_g, sigma200_b = corr_sigma(sigma200_s1, sigma200_g1, sigma200_b1, nmem_mass)

        err_sigma_s = e_sigma200_stat_s[notnan]
        err_sigma_g = e_sigma200_stat_g[notnan]
        err_sigma_b = e_sigma200_stat_b[notnan]

        log_err_sigma_s = e_sigma200_stat_s[notnan]/sigma200_s[notnan]/np.log(10)
        log_err_sigma_g = e_sigma200_stat_g[notnan]/sigma200_g[notnan]/np.log(10)
        log_err_sigma_b = e_sigma200_stat_b[notnan]/sigma200_b[notnan]/np.log(10)

        for im in [0]:
            im = fit_names[im]
            ssa = []
            ssb = []
            sga = []
            sgb = []
            sba = []
            sbb = []
            inss = []
            insg = []
            insb = []
            for ib in range(100):
                np.random.seed(1986*ib)
                vv = np.random.choice(range(len(M200[notnan])), len(M200[notnan]), replace=True)

                par_s1, per_s1, fit_s1 = linear_fit(mtest, M200[notnan][vv], sigma200_s[notnan][vv], err_sigma_s[vv], method=im)
                par_g1, per_g1, fit_g1 = linear_fit(mtest, M200[notnan][vv], sigma200_g[notnan][vv], err_sigma_g[vv], method=im)
                par_b1, per_b1, fit_b1 = linear_fit(mtest, M200[notnan][vv], sigma200_b[notnan][vv], err_sigma_b[vv], method=im)
                ssa.append(par_s1[0])
                ssb.append(par_s1[1])
                sga.append(par_g1[0])
                sgb.append(par_g1[1])
                sba.append(par_b1[0])
                sbb.append(par_b1[1])


                int_sca_boot_s = int_scat_est(par_s1, np.log10(M200[notnan][vv]), np.log10(sigma200_s[notnan][vv]), log_err_sigma_s[vv])
                int_sca_boot_g = int_scat_est(par_g1, np.log10(M200[notnan][vv]), np.log10(sigma200_g[notnan][vv]), log_err_sigma_g[vv])
                int_sca_boot_b = int_scat_est(par_b1, np.log10(M200[notnan][vv]), np.log10(sigma200_b[notnan][vv]), log_err_sigma_b[vv])
                inss.append(int_sca_boot_s)
                insg.append(int_sca_boot_g)
                insb.append(int_sca_boot_b)


            aux_ss = [np.mean(ssa),np.mean(ssb)]
            aux_ess = [np.std(ssa),np.std(ssb)]
            aux_sg = [np.mean(sga),np.mean(sgb)]
            aux_esg = [np.std(sga),np.std(sgb)]
            aux_sb = [np.mean(sba),np.mean(sbb)]
            aux_esb = [np.std(sba),np.std(sbb)]

            par_s.append(aux_ss)
            per_s.append(aux_ess)
            fit_s.append(fit_s1)

            par_g.append(aux_sg)
            per_g.append(aux_esg)
            fit_g.append(fit_g1)

            par_b.append(aux_sb)
            per_b.append(aux_esb)
            fit_b.append(fit_b1)


            sigmaint_s = np.mean(inss)
            e_sigmaint_s = np.std(inss)
            sigmaint_g = np.mean(insg)
            e_sigmaint_g = np.std(insg)
            sigmaint_b = np.mean(insb)
            e_sigmaint_b = np.std(insb)



        s_munari = munari13(mtest,tracer=tracers)
        s_saro = saro13(mtest)
        s_ho = ho19(mtest)

        residuals_s = (sigma200_s[notnan]- pow_law_func(M200[notnan], par_s[0][0], par_s[0][1]))
        log_residuals_s = np.log10(sigma200_s[notnan]/pow_law_func(M200[notnan], par_s[0][0], par_s[0][1]))
        residuals_g = (sigma200_g[notnan]- pow_law_func(M200[notnan], par_g[0][0], par_g[0][1]))
        log_residuals_g = np.log10(sigma200_g[notnan]/pow_law_func(M200[notnan], par_g[0][0], par_g[0][1]))
        residuals_b = (sigma200_b[notnan] - pow_law_func(M200[notnan], par_b[0][0], par_b[0][1]))
        log_residuals_b = np.log10(sigma200_b[notnan]/pow_law_func(M200[notnan], par_b[0][0], par_b[0][1]))


        print('n_clusters =',len(M200[notnan]))
        print('M200 = [',np.min(M200[notnan]),',', np.max(M200[notnan]),']')
        print("halo_mass =  10**(np.log10(M200[0]*1.e15))")
        print('std_'+mass_lim_name+', is_std_'+mass_lim_name+' = [',par_s[0],',', per_s[0],'] ,', sigmaint_s)
        print('gap_'+mass_lim_name+', is_gap_'+mass_lim_name+' = [',par_g[0],',', per_g[0],'] ,', sigmaint_g)
        print('bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+' = [',par_b[0],',', per_b[0],'] ,', sigmaint_b)
        print()
        if ngal_lim == 3:
            if i_h_mass_lim == 0: print('plot_par_mass_dep(ax1, colorVal['+str(i_s_mass_lim)+'], halo_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+', lab=r"$M_* = 10^{'+stell_mass_str+'}$")')
            else: print('plot_par_mass_dep(ax1, colorVal['+str(i_s_mass_lim)+'], halo_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+')')
            if i_s_mass_lim == 0: print('plot_par_mass_dep(ax2, colorVal2['+str(i_h_mass_lim)+'], stell_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+', lab=r"$M_{halo} = 10^{'+str(halo_mass_lim_names2[i_h_mass_lim])+'}$")')
            else: print('plot_par_mass_dep(ax2, colorVal2['+str(i_h_mass_lim)+'], stell_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+')')
        if ngal_lim >= 7:
            print('plot_par_mass_dep(ax1, colorVal['+str(i_s_mass_lim)+'], halo_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+", op='none')")
            print('plot_par_mass_dep(ax2, colorVal2['+str(i_h_mass_lim)+'], stell_mass, std_'+mass_lim_name+', is_std_'+mass_lim_name+', gap_'+mass_lim_name+', is_gap_'+mass_lim_name+', bwt_'+mass_lim_name+', is_bwt_'+mass_lim_name+", op='none')")


        pvalue_s = np.array(pvalue_s)
        pvalue_g = np.array(pvalue_g)
        pvalue_b = np.array(pvalue_b)
        kurt = np.array(kurt)
        skewn = np.array(skewn)
        col=[cluster_names, nmem, nmem_mass, M200, sigma200_s, err_sigma_s, sigma200_g, err_sigma_g, sigma200_b, err_sigma_b, residuals_s, residuals_g, residuals_b, log_residuals_s, log_residuals_g, log_residuals_b, pvalue_s[:,0], pvalue_s[:,1], pvalue_s[:,2], pvalue_g[:,0], pvalue_g[:,1], pvalue_g[:,2], pvalue_b[:,0], pvalue_b[:,1], pvalue_b[:,2], kurt[:,0], kurt[:,1], kurt[:,2], skewn[:,0], skewn[:,1], skewn[:,2]]
        names_col = ['cl_name', 'Ngal_tot', 'Ngal', 'M200', 'sigma200_s', 'e_sigma200_s', 'sigma200_g', 'e_sigma200_g', 'sigma200_b', 'e_sigma200_b', 'residuals_s', 'residuals_g', 'residuals_b', 'log_residuals_s', 'log_residuals_g', 'log_residuals_b', 'p_value_s_x', 'p_value_s_y', 'p_value_s_z', 'p_value_g_x', 'p_value_g_y', 'p_value_g_z', 'p_value_b_x', 'p_value_b_y', 'p_value_b_z', 'kurtosis_x', 'kurtosis_y', 'kurtosis_z', 'skeweness_x', 'skeweness_y', 'skeweness_z']
        data_table = Table(col, names=names_col,dtype=('S', 'i', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f','f'))
        data_table.write('/Users/antonioferragamo/Desktop/300_Clusters/Smass_dep/%s/fit_std_gap_bwt_%s_%s_%s_%s_%sgal.dat'%(tracers,tracers,red_dir,star_mass_strs,mass_lim_name,ngal_lim), format='ascii', delimiter=' ',overwrite=True)

        #RIMUOVERE UN TAB QUANDO SI VUOLE USARE
        Clusters_infos = Table()
        Clusters_infos['cl_name'] = cl_names
        Clusters_infos['M200'] = M200
        Clusters_infos['R200'] = R200
        Clusters_infos['df_fcl'] = np.array(df_fcl)
        Clusters_infos['df_fcl_norm_r200'] = np.array(df_fcl_r200)
        Clusters_infos['BCG_vel'] = np.array(BCG_vel)
        Clusters_infos['mean_vel'] = np.array(mean_vel)
        Clusters_infos['mean_vel_mass'] = np.array(mean_vel_mass)
        Clusters_infos['nmem'] = np.array(nmem)
        Clusters_infos['nmem_mass'] = np.array(nmem_mass)
        Clusters_infos['sigma200_s'] = sigma200_s
        Clusters_infos['err_sigma200_s'] = err_sigma_s
        Clusters_infos['sigma200_g'] = sigma200_g
        Clusters_infos['err_sigma200_g'] = err_sigma_g
        Clusters_infos['sigma200_b'] = sigma200_b
        Clusters_infos['err_sigma200_b'] = err_sigma_b
        Clusters_infos['residuals_s'] = residuals_s
        Clusters_infos['residuals_g'] = residuals_g
        Clusters_infos['residuals_b'] = residuals_b
        Clusters_infos['log_residuals_s'] = log_residuals_s
        Clusters_infos['log_residuals_g'] = log_residuals_g
        Clusters_infos['log_residuals_b'] = log_residuals_b
        Clusters_infos.write(file_path+'output/%s/clusters_info_vel_dist_%s_%s_%sgal%s.fits'%(tracers,stell_mass_str,mass_lim_name2,ngal_lim,non_com_str), overwrite=True)



"""
fig, ax = plt.subplots(3,3,figsize=(15, 8), gridspec_kw={'height_ratios':[3,1.5,1.5]},sharex=True)#,sharey=True)
fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)
fig.suptitle(tracers, fontsize=12)


ax[0,0].errorbar(M200[notnan], sigma200_g[notnan], yerr=err_sigma_s, ecolor='grey', elinewidth=2, color='g', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[0,0].loglog(mtest, fit_s[0],'-', color=fit_colors[0],label='300 Clusters',zorder=15)
ax[0,0].plot(mtest, s_munari, '-', color='magenta',label='Munari et al. (2013)')
ax[0,0].plot(mtest, s_saro, '-', color='darkorange',label='Saro et al. (2013)')
ax[0,0].plot(mtest, s_ho, '-', color='limegreen',label='Ho et al. (2019)[pure]')
ax[0,0].set_ylabel(r'$\sigma_{200}\,[km s^{-1}]$')
ax[0,0].set_title('Standard Deviation')

ax[1,0].set_ylabel('residuals')
residuals_s = (sigma200_s[notnan]- pow_law_func(M200[notnan], par_s[0][0], par_s[0][1]))
ax[1,0].errorbar(M200[notnan], residuals_s, yerr=err_sigma_s, ecolor='grey', elinewidth=2, color='g',fmt='o', mec='k',markersize=5, alpha=0.5)
ax[1,0].plot([1.e-2,5],[np.mean(residuals_s),np.mean(residuals_s)],'-', color='deepskyblue', zorder=10)
ax[1,0].plot([1.e-2,5],[np.mean(residuals_s)+np.std(residuals_s),np.mean(residuals_s)+np.std(residuals_s)],'--', color='deepskyblue', zorder=10)
ax[1,0].plot([1.e-2,5],[np.mean(residuals_s)-np.std(residuals_s),np.mean(residuals_s)-np.std(residuals_s)],'--', color='deepskyblue', zorder=10)

ax[2,0].set_ylabel('log(residuals)')
log_residuals_s = np.log10(sigma200_s[notnan]/pow_law_func(M200[notnan], par_s[0][0], par_s[0][1]))
ax[2,0].errorbar(M200[notnan], log_residuals_s, yerr=log_err_sigma_s, ecolor='grey', elinewidth=2, color='g',fmt='o', mec='k',markersize=5, alpha=0.5)
ax[2,0].plot([1.e-2,5],[np.mean(log_residuals_s),np.mean(log_residuals_s)],'-', color='deepskyblue', zorder=10)
ax[2,0].plot([1.e-2,5],[np.mean(log_residuals_s)+np.std(log_residuals_s),np.mean(log_residuals_s)+np.std(log_residuals_s)],'--', color='deepskyblue', zorder=10)
ax[2,0].plot([1.e-2,5],[np.mean(log_residuals_s)-np.std(log_residuals_s),np.mean(log_residuals_s)-np.std(log_residuals_s)],'--', color='deepskyblue', zorder=10)


ax[2,1].set_xlabel('$M_{200}\\,[10^{15} h^{-1} M_{\\odot}]$')

ax[0,1].errorbar(M200[notnan], sigma200_g[notnan], yerr=err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[0,1].loglog(mtest, fit_g[0],'-', color=fit_colors[0],label="300 Clusters",zorder=15)
ax[0,1].plot(mtest, s_munari, '-', color='magenta',label='Munari et al. (2013)')
ax[0,1].plot(mtest, s_saro, '-', color='darkorange',label='Saro et al. (2013)')
ax[0,1].plot(mtest, s_ho, '-', color='limegreen',label='Ho et al. (2019)[pure]')
ax[0,1].set_title('Gapper')

residuals_g = (sigma200_g[notnan]- pow_law_func(M200[notnan], par_g[0][0], par_g[0][1]))
ax[1,1].errorbar(M200[notnan], residuals_g, yerr=err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[1,1].plot([1.e-2,5],[np.mean(residuals_g),np.mean(residuals_g)],'-', color='deepskyblue', zorder=10)
ax[1,1].plot([1.e-2,5],[np.mean(residuals_g)+np.std(residuals_g),np.mean(residuals_g)+np.std(residuals_g)],'--', color='deepskyblue', zorder=10)
ax[1,1].plot([1.e-2,5],[np.mean(residuals_g)-np.std(residuals_g),np.mean(residuals_g)-np.std(residuals_g)],'--', color='deepskyblue', zorder=10)

log_residuals_g = np.log10(sigma200_g[notnan]/pow_law_func(M200[notnan], par_g[0][0], par_g[0][1]))
ax[2,1].errorbar(M200[notnan], log_residuals_g, yerr=log_err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[2,1].plot([1.e-2,5],[np.mean(log_residuals_g),np.mean(log_residuals_g)],'-', color='deepskyblue', zorder=10)
ax[2,1].plot([1.e-2,5],[np.mean(log_residuals_g)+np.std(log_residuals_g),np.mean(log_residuals_g)+np.std(log_residuals_g)],'--', color='deepskyblue', zorder=10)
ax[2,1].plot([1.e-2,5],[np.mean(log_residuals_g)-np.std(log_residuals_g),np.mean(log_residuals_g)-np.std(log_residuals_g)],'--', color='deepskyblue', zorder=10)

ax[0,2].errorbar(M200[notnan], sigma200_b[notnan], yerr=err_sigma_b, ecolor='grey', elinewidth=2, color='b', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[0,2].loglog(mtest, fit_b[0],'-', color=fit_colors[0],label='300 Clusters',zorder=15)
ax[0,2].plot(mtest, s_munari, '-', color='magenta',label='Munari et al. (2013)')
ax[0,2].plot(mtest, s_saro, '-', color='darkorange',label='Saro et al. (2013)')
ax[0,2].plot(mtest, s_ho, '-', color='limegreen',label='Ho et al. (2019)[pure]')
ax[0,2].set_title('Biweight')
ax[0,2].get_yaxis().set_visible(False)
residuals_b = (sigma200_b[notnan] - pow_law_func(M200[notnan], par_b[0][0], par_b[0][1]))
ax[1,2].errorbar(M200[notnan], residuals_b, yerr=err_sigma_b, ecolor='grey', elinewidth=2, color='b', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[1,2].plot([1.e-2,5],[np.mean(residuals_b),np.mean(residuals_b)],'-', color='deepskyblue', zorder=10)
ax[1,2].plot([1.e-2,5],[np.mean(residuals_b)+np.std(residuals_b),np.mean(residuals_b)+np.std(residuals_b)],'--', color='deepskyblue', zorder=10)
ax[1,2].plot([1.e-2,5],[np.mean(residuals_b)-np.std(residuals_b),np.mean(residuals_b)-np.std(residuals_b)],'--', color='deepskyblue', zorder=10)

log_residuals_b1 = np.log10(sigma200_b[notnan]/pow_law_func(M200[notnan], par_b[0][0], par_b[0][1]))
ax[2,2].errorbar(M200[notnan], log_residuals_b1, yerr=log_err_sigma_b, ecolor='grey', elinewidth=2, color='b', fmt='o', mec='k',markersize=5, alpha=0.5)
ax[2,2].plot([1.e-2,5],[np.mean(log_residuals_b1),np.mean(log_residuals_b1)],'-', color='deepskyblue', zorder=10)
ax[2,2].plot([1.e-2,5],[np.mean(log_residuals_b1)+np.std(log_residuals_b1),np.mean(log_residuals_b1)+np.std(log_residuals_b1)],'--', color='deepskyblue', zorder=10)
ax[2,2].plot([1.e-2,5],[np.mean(log_residuals_b1)-np.std(log_residuals_b1),np.mean(log_residuals_b1)-np.std(log_residuals_b1)],'--', color='deepskyblue', zorder=10)



fig, ax2 = plt.subplots(3,1,figsize=(8, 8), gridspec_kw={'height_ratios':[3,1.5,1.5]},sharex=True)#,sharey=True)
fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)

ax2[0].errorbar(M200[notnan], sigma200_g[notnan], yerr=err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5,zorder=1)
ax2[0].set_xscale('log')
ax2[0].set_yscale('log')
for ifit in range(1):
    rcL = []
    a,b = par_g[ifit][0], par_g[ifit][1]
    e_a, e_b = per_g[ifit][0], per_g[ifit][1]
    for i in range(0,10000):
        aa = np.random.normal(a,e_a)
        bb = np.random.normal(b,e_b)
        ff = pow_law_func(mtest, aa, bb)
        rcL = np.append(rcL, ff)
    rcL = rcL.reshape(10000, len(mtest))
    err_fit_gap = np.nanstd(rcL, axis=0)
    fit_gap = np.nanmean(rcL, axis=0)

    ax2[0].loglog(mtest, fit_gap,'-', color=fit_colors[ifit],label='300 Clusters [Hydro]')
    fitm = fit_gap - err_fit_gap
    fitp = fit_gap + err_fit_gap
    ax2[0].fill_between(mtest, fitm, fitp, color=fit_colors[ifit], alpha =0.6,zorder=2)
ax2[0].legend(loc='lower right',frameon=False,fontsize=8,markerfirst=False)
ax2[0].axis([3.e-2,5,1.5e2,2e3])
ax2[0].set_ylabel(r'$\sigma_{200}\,[km s^{-1}]$')
ax2[0].get_yaxis().set_visible(True)

ax2[1].set_ylabel('residuals')
ax2[1].errorbar(M200[notnan], residuals_g, yerr=err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5)
ax2[1].plot([1.e-2,5],[np.mean(residuals_g),np.mean(residuals_g)],'-', color='deepskyblue', zorder=10)
ax2[1].plot([1.e-2,5],[np.mean(residuals_g)+np.std(residuals_g),np.mean(residuals_g)+np.std(residuals_g)],'--', color='deepskyblue', zorder=10)
ax2[1].plot([1.e-2,5],[np.mean(residuals_g)-np.std(residuals_g),np.mean(residuals_g)-np.std(residuals_g)],'--', color='deepskyblue', zorder=10)

ax2[1].axis([1.5e-2,5,-650,650])

ax2[2].set_ylabel('log(residuals)')
ax2[2].errorbar(M200[notnan], log_residuals_g, yerr=log_err_sigma_g, ecolor='grey', elinewidth=2, color='r', fmt='o', mec='k',markersize=5, alpha=0.5)
ax2[2].plot([1.e-2,5],[np.mean(log_residuals_g),np.mean(log_residuals_g)],'-', color='deepskyblue', zorder=10)
ax2[2].plot([1.e-2,5],[np.mean(log_residuals_g)+np.std(log_residuals_g),np.mean(log_residuals_g)+np.std(log_residuals_g)],'--', color='deepskyblue', zorder=10)
ax2[2].plot([1.e-2,5],[np.mean(log_residuals_g)-np.std(log_residuals_g),np.mean(log_residuals_g)-np.std(log_residuals_g)],'--', color='deepskyblue', zorder=10)

ax2[2].set_xlabel('$M_{200}\\,[10^{15} h^{-1}M_{\\odot}]$')
"""
