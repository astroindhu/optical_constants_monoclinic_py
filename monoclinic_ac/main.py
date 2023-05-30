import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import shutil
import os
from genetic_algorithm_modules import ga_optical_constants, chisq, func_lsq_R
from dispersion_fresnel_modules import calculate_rnk
from input_data import r, v, p, viewing_angle, v1, r_omega1, r_omega2, r_omega3, r_omega4

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2


L = len(v1)

spectra_range = {'R1': range(32), 'R2': range(32,81), 'R3':range(81, 116), 'R4':range(116, 210), 'R5':range(210, 359), 'R6':range(359, 405),  'R7': range(405, len(v1))}

def count(sett, rangee):
    c = 0
    for val in sett:
        if rangee[0] < val < rangee[-1]:
            c += 1
    return c

def chisq_val_Rn(rfit, Rn):
    rfit_omega1 = rfit[:L]
    rfit_omega2 = rfit[L:2*L]
    rfit_omega3 = rfit[2*L:3*L]
    rfit_omega4 = rfit[3*L:4*L]

    chisq_v = (chisq(rfit_omega1[spectra_range[Rn]], r_omega1[spectra_range[Rn]]) +
                   chisq(rfit_omega2[spectra_range[Rn]], r_omega2[spectra_range[Rn]]) +
                   chisq(rfit_omega3[spectra_range[Rn]], r_omega3[spectra_range[Rn]]) +
                   chisq(rfit_omega4[spectra_range[Rn]], r_omega4[spectra_range[Rn]]))/4

    return chisq_v



# folder for saving the results for monoclinic solution ac

path_full = "/Users/astroindhu/SBU/1_Optical_Constants/optical_constants_py/MIR_optical_constants_py/monoclinic_py/monoclinic_py/monoclinic_ac/"
lf = path_full + 'py_solution_ac/'

if os.path.isdir(lf):
    shutil.rmtree(lf)
    os.mkdir(lf)
else:
    os.mkdir(lf)

max_num_oscillators_R1 = 1
max_num_oscillators_R2 = 5
max_num_oscillators_R3 = 2
max_num_oscillators_R4 = 5
max_num_oscillators_R5 = 5
max_num_oscillators_R6 = 5
max_num_oscillators_R7 = 2

num_oscillators = 2
total_num_oscillators = max_num_oscillators_R1 + max_num_oscillators_R2 + max_num_oscillators_R3 + max_num_oscillators_R4 + max_num_oscillators_R5 + max_num_oscillators_R6
run = 1

while num_oscillators < total_num_oscillators:
    if run == 1:
        lb_ga_nu = np.repeat(400, num_oscillators)
        ub_ga_nu = np.repeat(1200, num_oscillators)

    else:
        lb_ga_nu = ga_solution[:num_oscillators] - 10.
        ub_ga_nu = ga_solution[:num_oscillators] + 10.


        if chisq_val_R1 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R1']]) < max_num_oscillators_R1:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R1']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R1']][-1]-10)

        if chisq_val_R2 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R2']]) < max_num_oscillators_R2:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R2']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R2']][-1]-10)

        if chisq_val_R3 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R3']]) < max_num_oscillators_R3:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R3']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R3']][-1]-10)

        if chisq_val_R4 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R4']]) < max_num_oscillators_R4:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R4']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R4']][-1]-10)

        if chisq_val_R5 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R5']]) < max_num_oscillators_R5:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R5']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R5']][-1]-10)

        if chisq_val_R6 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R6']]) < max_num_oscillators_R6:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R6']][0]+10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R6']][-1]-10)

        if chisq_val_R7 >= 0.03:
            if count(ga_solution[:num_oscillators], v1[spectra_range['R7']]) < max_num_oscillators_R7:
                lb_ga_nu = np.append(lb_ga_nu, v1[spectra_range['R7']][0] + 10)
                ub_ga_nu = np.append(ub_ga_nu, v1[spectra_range['R7']][-1] - 10)

    num_oscillators = len(lb_ga_nu)

    print("run = %s ; num_oscillators = %s" % (run, num_oscillators))

    lb_ga_nu = np.sort(lb_ga_nu)
    ub_ga_nu = np.sort(ub_ga_nu)

    lb_ga_gamm = np.repeat(0, num_oscillators)
    ub_ga_gamm = np.repeat(100, num_oscillators)

    lb_ga_Sk = np.repeat(1000, num_oscillators)
    ub_ga_Sk = np.repeat(1000000, num_oscillators)

    lb_ga_phi = np.repeat(-180, num_oscillators)
    ub_ga_phi = np.repeat(180, num_oscillators)

    # lb_ga_theta = np.repeat(89.9, num_oscillators)
    # ub_ga_theta = np.repeat(90.1, num_oscillators)
    #
    # lb_ga = np.hstack((lb_ga_nu, lb_ga_gamm, lb_ga_Sk, lb_ga_phi, lb_ga_theta))  # lower bound of genes representing gamm, Sk
    # ub_ga = np.hstack((ub_ga_nu, ub_ga_gamm, ub_ga_Sk, ub_ga_phi, ub_ga_theta))  # upper bound of genes representing gamm, Sk


    lb_ga = np.hstack((lb_ga_nu, lb_ga_gamm, lb_ga_Sk, lb_ga_phi))  # lower bound of genes representing gamm, Sk
    ub_ga = np.hstack((ub_ga_nu, ub_ga_gamm, ub_ga_Sk, ub_ga_phi))  # upper bound of genes representing gamm, Sk

    gene_space = []
    for i in np.arange(len(lb_ga)):
        gene_space.append({'low': lb_ga[i], 'high': ub_ga[i]})

    for i in range(num_oscillators):
        gene_space.append([90])

    gene_space.append({'low': 0, 'high': 5})  # epsilxx
    gene_space.append({'low': 0, 'high': 5})  # epsilxy
    gene_space.append({'low': 0, 'high': 5})  # epsilyy
    gene_space.append({'low': 0, 'high': 5})  # epsilzz

    print("Initiating GA routine")

    ga_instance = ga_optical_constants(gene_space)
    ga_solution, ga_solution_fitness, ga_solution_idx = ga_instance.best_solution()
    #     ga_instance.plot_fitness()

    rfit_ga, n1_ga, k1_ga, n2_ga, k2_ga = calculate_rnk(ga_solution, p, viewing_angle)

    print("Completed GA routine")

    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=ga_solution_fitness))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    ga_chisq = {}
    ga_chisq["overall_chisq"] = chisq(rfit_ga, r)
    print("Chi-square goodness of overall fitness is ", chisq(rfit_ga, r))

    chisq_val_R1 = chisq_val_Rn(rfit_ga,'R1')
    chisq_val_R2 = chisq_val_Rn(rfit_ga, 'R2')
    chisq_val_R3 = chisq_val_Rn(rfit_ga, 'R3')
    chisq_val_R4 = chisq_val_Rn(rfit_ga, 'R4')
    chisq_val_R5 = chisq_val_Rn(rfit_ga, 'R5')
    chisq_val_R6 = chisq_val_Rn(rfit_ga, 'R6')
    chisq_val_R7 = chisq_val_Rn(rfit_ga, 'R7')

    n1_ga = n1_ga[:L]
    n2_ga = n2_ga[:L]
    k1_ga = k1_ga[:L]
    k2_ga = k2_ga[:L]

# start of LSQ routine
    ga_oscillation_parameters = np.copy(ga_solution)

    N = int((len(ga_oscillation_parameters)- 4) / 5)

    nu = ga_oscillation_parameters[0:N]
    gamm = ga_oscillation_parameters[N:2 * N]
    Sk = ga_oscillation_parameters[2 * N:3 * N]
    phi = ga_oscillation_parameters[3 * N:4 * N]
    theta = ga_oscillation_parameters[4 * N:5 * N]
    epsilxx = ga_oscillation_parameters[5 * N]
    epsilxy = ga_oscillation_parameters[(5 * N) + 1]
    epsilyy = ga_oscillation_parameters[(5 * N) + 2]
    epsilzz = ga_oscillation_parameters[(5 * N) + 3]

    ga_oscillators_initial = np.hstack((nu, gamm, Sk, phi, theta, epsilxx, epsilxy, epsilyy, epsilzz))

    # gamm=gamm./nu;
    # fourpr=Sk./(nu.^2);

    # Set lower and upper bounds
    lb_nu = -20. + nu
    ub_nu = 20. + nu
    lb_gamm = np.tile(0, (1, len(nu)))
    ub_gamm = np.tile(100, (1, len(nu)))
    lb_Sk = np.tile(1000, (1, len(nu)))
    ub_Sk = np.tile(1000000, (1, len(nu)))
    lb_phi = np.tile(-180, (1, len(nu)))
    ub_phi = np.tile(180, (1, len(nu)))
    lb_theta = np.tile(89, (1, len(nu)))
    ub_theta = np.tile(91, (1, len(nu)))

    lb = np.hstack((lb_nu, lb_gamm.flatten(), lb_Sk.flatten(), lb_phi.flatten(), lb_theta.flatten(), 0, 0, 0, 0))
    ub = np.hstack((ub_nu, ub_gamm.flatten(), ub_Sk.flatten(), ub_phi.flatten(), ub_theta.flatten(), 5, 5, 5, 5))

    print("Initiating LSQ routine")

    ga_oscillators_final, lsq_cov = curve_fit(func_lsq_R, xdata=v, p0=ga_oscillators_initial, ydata=r, bounds=(lb, ub),
                                              maxfev=10000)

    print("Completed LSQ routine")

    rfit_lsq, n1_lsq, k1_lsq, n2_lsq, k2_lsq = calculate_rnk(ga_oscillators_final, p, viewing_angle)


    nu_final = ga_oscillators_final[0:N]
    gamm_final = ga_oscillators_final[N:2 * N]
    Sk_final = ga_oscillators_final[2 * N:3 * N]
    phi_final = ga_oscillators_final[3 * N:4 * N]
    theta_final = ga_oscillators_final[4 * N:5 * N]
    epsilxx_final = ga_oscillators_final[5 * N]
    epsilxy_final = ga_oscillators_final[(5 * N) + 1]
    epsilyy_final = ga_oscillators_final[(5 * N) + 2]
    epsilzz_final = ga_oscillators_final[(5 * N) + 3]


    chisq_val_R1 = chisq_val_Rn(rfit_lsq, 'R1')
    chisq_val_R2 = chisq_val_Rn(rfit_lsq, 'R2')
    chisq_val_R3 = chisq_val_Rn(rfit_lsq, 'R3')
    chisq_val_R4 = chisq_val_Rn(rfit_lsq, 'R4')
    chisq_val_R5 = chisq_val_Rn(rfit_lsq, 'R5')
    chisq_val_R6 = chisq_val_Rn(rfit_lsq, 'R6')
    chisq_val_R7 = chisq_val_Rn(rfit_lsq, 'R7')

    n1_lsq = n1_lsq[:L]
    n2_lsq = n2_lsq[:L]
    k1_lsq = k1_lsq[:L]
    k2_lsq = k2_lsq[:L]

    fname = lf+"ga_lsq_run"+str(run)+"_N"+str(num_oscillators)


    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    # ax[0].plot(v1, n1_ga,  'g', label='n1_ga')
    ax[0].plot(v1, n1_lsq, 'b', label='n1_ga+lsq')
    ax[0].plot(v1, n2_lsq, 'b', label='n2_ga+lsq')
    ax[0].set_xlim(2000, 400)
    ax[0].legend(bbox_to_anchor=(1, 1))
    ax[0].set_title('GA+LSQ modelled n2')

    ax[1].plot(v1, k1_lsq, 'b', label='k1_ga+lsq')
    ax[1].plot(v1, k2_lsq, 'b', label='k2_ga+lsq')
    ax[1].set_xlim(2000, 400)
    ax[1].legend(bbox_to_anchor=(1, 1))
    ax[1].set_title('GA+LSQ modelled k2')
    ax[1].set_yscale('log')
    plt.legend()
    fig.savefig(fname+'_k.png', bbox_inches="tight")
    plt.show()

    rfit_ga_omega1 = rfit_ga[:L]
    rfit_ga_omega2 = rfit_ga[L:2*L]
    rfit_ga_omega3 = rfit_ga[2*L:3*L]
    rfit_ga_omega4 = rfit_ga[3*L:4*L]

    rfit_lsq_omega1 = rfit_lsq[:L]
    rfit_lsq_omega2 = rfit_lsq[L:2*L]
    rfit_lsq_omega3 = rfit_lsq[2*L:3*L]
    rfit_lsq_omega4 = rfit_lsq[3*L:4*L]



    fig, ax = plt.subplots(4, 1, figsize=(17, 20))

    ax[0].plot(v1, rfit_lsq_omega1, 'b', label='R_ga+lsq')
    # ax[0].plot(v1, rfit_ga_omega1, 'r', label='R_ga')
    ax[0].plot(v1, r_omega1, 'k', label='R')
    ax[0].set_xlim(2000, 400)
    ax[0].legend(bbox_to_anchor=(1, 1))
    ax[0].set_title('GA+LSQ modelled R 0 deg')

    ax[1].plot(v1, rfit_lsq_omega2, 'b', label='R_ga+lsq')
    # ax[1].plot(v1, rfit_ga_omega2, 'r', label='R_ga')
    ax[1].plot(v1, r_omega2, 'k', label='R')
    ax[1].set_xlim(2000, 400)
    ax[1].legend(bbox_to_anchor=(1, 1))
    ax[1].set_title('GA+LSQ modelled R 45 deg')

    ax[2].plot(v1, rfit_lsq_omega3, 'b', label='R_ga+lsq')
    # ax[2].plot(v1, rfit_ga_omega3, 'r', label='R_ga')
    ax[2].plot(v1, r_omega3, 'k', label='R')
    ax[2].set_xlim(2000, 400)
    ax[2].legend(bbox_to_anchor=(1, 1))
    ax[2].set_title('GA+LSQ modelled R 90 deg')

    ax[3].plot(v1, rfit_lsq_omega4, 'b', label='R_ga+lsq')
    # ax[3].plot(v1, rfit_ga_omega4, 'r', label='R_ga')
    ax[3].plot(v1, r_omega4, 'k', label='R')
    ax[3].set_xlim(2000, 400)
    ax[3].legend(bbox_to_anchor=(1, 1))
    ax[3].set_title('GA+LSQ modelled R 135 deg')
    plt.legend()
    fig.savefig(fname+'_R.png', bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(4, 1, figsize=(17, 20))
    ax[0].plot(v1, r_omega1 - rfit_lsq_omega1, 'k')
    ax[0].set_title('GA+LSQ modelled R 0 deg')
    ax[0].set_ylim(-0.05, +0.05)
    ax[0].set_xlim(2000, 400)

    ax[1].plot(v1, r_omega2 - rfit_lsq_omega2, 'k')
    ax[1].set_title('GA+LSQ modelled R 45 deg')
    ax[1].set_ylim(-0.05, +0.05)
    ax[1].set_xlim(2000, 400)

    ax[2].plot(v1, r_omega3 - rfit_lsq_omega3, 'k')
    ax[2].set_title('GA+LSQ modelled R 90 deg')
    ax[2].set_ylim(-0.05, +0.05)
    ax[2].set_xlim(2000, 400)

    ax[3].plot(v1, r_omega4 - rfit_lsq_omega4, 'k')
    ax[3].set_title('GA+LSQ modelled R 135 deg')
    ax[3].set_ylim(-0.05, +0.05)
    ax[3].set_xlim(2000, 400)
    plt.legend()
    fig.savefig(fname+'_res.png', bbox_inches="tight")
    plt.show()


    ga_lsq_chisq = {}

    chisq_val = chisq(r[1:], rfit_lsq[1:])


    print("Chi-square goodness of fitness is ", chisq_val)

    print("Chi-square goodness of R1 fitness is ", chisq_val_R1)
    print("Chi-square goodness of R2 fitness is ", chisq_val_R2)
    print("Chi-square goodness of R3 fitness is ", chisq_val_R3)
    print("Chi-square goodness of R4 fitness is ", chisq_val_R4)
    print("Chi-square goodness of R5 fitness is ", chisq_val_R5)
    print("Chi-square goodness of R6 fitness is ", chisq_val_R6)
    print("Chi-square goodness of R7 fitness is ", chisq_val_R7)

    fig, ax2 = plt.subplots(1, 2, figsize=(10, 3))

    ax2[0].plot(v1, r_omega1)
    for i in ga_solution[:num_oscillators]:
        ax2[0].scatter(i, r_omega1[np.abs(v1 - i).argmin()], color='r')
        # ax2[0].scatter(i, r_omega1[np.abs(v1 - i).argmin()], label=round(i, 3), color='r')
    # ax2[0].legend()
    ax2[0].set_title("GA oscillators")
    ax2[0].set_xlim(2000, 400)

    ax2[1].plot(v1, r_omega1)
    for i in ga_oscillators_final[:num_oscillators]:
        ax2[1].scatter(i, r_omega1[np.abs(v1 - i).argmin()], color='r')
        # ax2[1].scatter(i, r_omega1[np.abs(v1 - i).argmin()], label=round(i, 3), color='r')
    # ax2[1].legend()
    ax2[1].set_title("GA+LSQ oscillators")
    ax2[1].set_xlim(2000, 400)
    fig.savefig(fname + '_osc.png', bbox_inches="tight")

    plt.show()

    R_residuals = r - rfit_lsq


    #save oscillator parameters
    # save_oscillator_parameters(fname, ga_oscillators_final, ga_lsq_chisq)

    df = pd.DataFrame(np.stack((nu_final,
                                gamm_final,
                                Sk_final,
                                phi_final,
                                theta_final,
                                np.tile(epsilxx_final,N),
                                np.tile(epsilxy_final,N),
                                np.tile(epsilyy_final,N),
                                np.tile(epsilzz_final,N)), axis=1),
                      columns = ["nu",
                                 "gamm",
                                 "Sk",
                                 "phi",
                                 "theta",
                                 "epsilxx",
                                 "epsilxy",
                                 "epsilyy",
                                 "epsilzz"])
    df.to_excel(fname+"_oscillator_parameters.xlsx")


    #save optical constants to excel
    df = pd.DataFrame(np.stack((v1,
                                r_omega1,
                                rfit_lsq_omega1,
                                r_omega2,
                                rfit_lsq_omega2,
                                r_omega3,
                                rfit_lsq_omega3,
                                r_omega4,
                                rfit_lsq_omega4,
                                n1_lsq.flatten(),
                                n2_lsq.flatten(),
                                k1_lsq.flatten(),
                                k2_lsq.flatten()), axis=1),
                      columns= ["wavenumber",
                                "r_omega1",
                                "r_modelled_omega1",
                                "r_omega2",
                                "r_modelled_omega2",
                                "r_omega3",
                                "r_modelled_omega3",
                                "r_omega4",
                                "r_modelled_omega4",
                                "n1" ,
                                "n2",
                                "k1",
                                "k2"])
    df.to_excel(fname+"_optical_constants.xlsx")

    dict_chi = {'run': [run],
                'N': [N],
                'chisq_all': [chisq_val],
                'chisq_R1': [chisq_val_R1],
                'chisq_R2': [chisq_val_R2],
                'chisq_R3': [chisq_val_R3],
                'chisq_R4': [chisq_val_R4],
                'chisq_R5': [chisq_val_R5],
                'chisq_R6': [chisq_val_R6],
                'chisq_R7': [chisq_val_R7]
                }

    if run == 1:
        df_chi = pd.DataFrame(dict_chi)
    else:
        df_chi = df_chi.append(dict_chi, ignore_index=True)

    run += 1

    if (N >= total_num_oscillators) or (run >= total_num_oscillators):
        break


df_chi.to_excel('chisq_all_runs.xlsx')

