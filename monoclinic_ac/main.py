import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shutil
import os
from genetic_algorithm_modules import ga_optical_constants
from dispersion_fresnel_modules import calculate_rnk
from input_data import r, v, viewing_angle


def count(sett, rangee):
    c = 0
    for val in sett:
        if rangee[0] < val < rangee[-1]:
            c += 1
    return c

# folder for saving the results for monoclinic solution ac

path_full = "/Users/astroindhu/SBU/1_Optical_Constants/optical_constants_py/MIR_optical_constants_py/monoclinic_py/MIR_Monoclinic_Optical_Constants_Code/monoclinic_v2/"
lf = path_full + 'py_solution_ac/'

if os.path.isdir(lf):
    shutil.rmtree(lf)
    os.mkdir(lf)
else:
    os.mkdir(lf)

# max_num_oscillators_R1 = 1
# max_num_oscillators_R2 = 2
# max_num_oscillators_R3 = 2
# max_num_oscillators_R4 = 4
# max_num_oscillators_R5 = 5
# max_num_oscillators_R6 = 5
# max_num_oscillators_R7 = 0

osc2 = 1
num_oscillators = 2
total_num_oscillators = 13
# total_num_oscillators = max_num_oscillators_R1 + max_num_oscillators_R2 + max_num_oscillators_R3 + max_num_oscillators_R4 + max_num_oscillators_R5 + max_num_oscillators_R6
run = 1

while num_oscillators < total_num_oscillators:
    if osc2 == 1:
        lb_ga_nu = np.repeat(420, num_oscillators)
        ub_ga_nu = np.repeat(1300, num_oscillators)
        osc2 = 0

    else:
        lb_ga_nu = ga_solution[:num_oscillators] - 10.
        ub_ga_nu = ga_solution[:num_oscillators] + 10.

        if chisq_val >= 0.03:
            lb_ga_nu = np.append(lb_ga_nu, 420)
            ub_ga_nu = np.append(ub_ga_nu, 1300)

        #         if chisq_val_R1 >= 0.03:
    #             lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R1']][0]+10)
    #             ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R1']][-1]-10)

    #         if chisq_val_R2 >= 0.001:
    #             if count(ga_solution[:num_oscillators], v[spectra_range['R2']]) < max_num_oscillators_R2:
    #                 lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R2']][0]+10)
    #                 ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R2']][-1]-10)

    #         if chisq_val_R3 >= 0.001:
    #             if count(ga_solution[:num_oscillators], v[spectra_range['R3']]) < max_num_oscillators_R3:
    #                 lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R3']][0]+10)
    #                 ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R3']][-1]-10)

    #         if chisq_val_R4 >= 0.001:
    #             if count(ga_solution[:num_oscillators], v[spectra_range['R4']]) < max_num_oscillators_R4:
    #                 lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R4']][0]+10)
    #                 ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R4']][-1]-10)

    #         if chisq_val_R5 >= 0.001:
    #             if count(ga_solution[:num_oscillators], v[spectra_range['R5']]) < max_num_oscillators_R5:
    #                 lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R5']][0]+10)
    #                 ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R5']][-1]-10)

    #         if chisq_val_R6 >= 0.001:
    #             if count(ga_solution[:num_oscillators], v[spectra_range['R6']]) < max_num_oscillators_R6:
    #                 lb_ga_nu = np.append(lb_ga_nu, v[spectra_range['R6']][0]+10)
    #                 ub_ga_nu = np.append(ub_ga_nu, v[spectra_range['R6']][-1]-10)

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

    lb_ga_theta = np.repeat(89, num_oscillators)
    ub_ga_theta = np.repeat(91, num_oscillators)

    lb_ga = np.hstack((lb_ga_nu, lb_ga_gamm, lb_ga_Sk))  # lower bound of genes representing gamm, Sk
    ub_ga = np.hstack((ub_ga_nu, ub_ga_gamm, ub_ga_Sk))  # upper bound of genes representing gamm, Sk

    gene_space = []
    for i in np.arange(len(lb_ga)):
        gene_space.append({'low': lb_ga[i], 'high': ub_ga[i]})
    gene_space.append({'low': 0, 'high': 5})  # epsilxx
    gene_space.append({'low': 0, 'high': 5})  # epsilxy
    gene_space.append({'low': 0, 'high': 5})  # epsilyy
    gene_space.append({'low': 0, 'high': 5})  # epsilzz

    ga_instance = ga_optical_constants(gene_space)
    ga_solution, ga_solution_fitness, ga_solution_idx = ga_instance.best_solution()
    #     ga_instance.plot_fitness()

    rfit_ga, n1_ga, k1_ga, n2_ga, k2_ga = calculate_rnk(ga_solution, v, viewing_angle)

    #     fig, ax = plt.subplots(1,4, figsize=(15, 3))
    #     ax[0].plot(v,n_ga,'r', label='n_ga')
    #     ax[0].legend(bbox_to_anchor=(1,1))
    #     ax[0].set_title('GA modelled n')

    #     ax[1].plot(v,k_ga,'g', label='k_ga')
    #     ax[1].legend(bbox_to_anchor=(1,1))
    #     ax[1].set_title('GA modelled k')

    #     ax[2].plot(v,R_ga,'b', label='R_ga')
    #     ax[2].plot(v,R_smoothed,'k', label='R')
    #     ax[2].legend(bbox_to_anchor=(1,1))
    #     ax[2].set_title('GA modelled R')

    #     ax[3].plot(v, R_smoothed-R_ga, label='residual R')
    #     ax[3].set_ylim(-0.05, +0.05)
    #     plt.show()

    #     print("\t".join(map(str, ga_solution)))

    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=ga_solution_fitness))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    ga_chisq = {}
    ga_chisq["overall_chisq"] = chisq(rfit_ga, r)
    print("Chi-square goodness of overall fitness is ", chisq(rfit_ga, r))

    #     ga_chisq["chisq_val_R1"] = chisq(R_ga[spectra_range['R1']], R_smoothed[spectra_range['R1']])
    #     ga_chisq["chisq_val_R2"] = chisq(R_ga[spectra_range['R2']], R_smoothed[spectra_range['R2']])
    #     ga_chisq["chisq_val_R3"] = chisq(R_ga[spectra_range['R3']], R_smoothed[spectra_range['R3']])
    #     ga_chisq["chisq_val_R4"] = chisq(R_ga[spectra_range['R4']], R_smoothed[spectra_range['R4']])
    #     ga_chisq["chisq_val_R5"] = chisq(R_ga[spectra_range['R5']], R_smoothed[spectra_range['R5']])
    #     ga_chisq["chisq_val_R6"] = chisq(R_ga[spectra_range['R6']], R_smoothed[spectra_range['R6']])
    #     ga_chisq["chisq_val_R7"] = chisq(R_ga[spectra_range['R7']], R_smoothed[spectra_range['R7']])

    #     np.savetxt(lf+'/run%s_%s_NumOfOscillators_ga_best_solution_fitness_per_generation.txt'%(run, num_oscillators), ga_instance.best_solutions_fitness)
    #     fname = lf+"ga_run"+str(run)+"_Noscillators"+str(num_oscillators)
    #     #save oscillator parameters to excel
    #     save_oscillator_parameters(fname, ga_solution, ga_chisq)
    #     #save optical constants to excel
    #     df = pd.DataFrame(np.stack((v, R_smoothed, R_ga, R_smoothed-R_ga, n_ga, k_ga), axis=1), columns= ["wavenumber", "R_measured", "R_modelled", "R_residuals", "n" , "k"] )
    #     df.to_excel(fname+"_optical_constants.xlsx")

    ga_oscillation_parameters = np.copy(ga_solution)

    N = int((len(ga_oscillation_parameters) - 1) / 3)

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
    lb_gamm = np.zeros((0, len(nu)))
    ub_gamm = np.tile(100, (1, len(nu)))
    lb_Sk = np.zeros((1000, len(nu)))
    ub_Sk = np.tile(1000000, (1, len(nu)))
    lb_phi = np.zeros((-180, len(nu)))
    ub_phi = np.tile(180, (1, len(nu)))
    lb_theta = np.zeros((89, len(nu)))
    ub_theta = np.tile(91, (1, len(nu)))

    lb = np.hstack((lb_nu, lb_gamm.flatten(), lb_Sk.flatten(), lb_phi.flatten(), lb_theta.flatten(), 0, 0, 0, 0))
    ub = np.hstack((ub_nu, ub_gamm.flatten(), ub_Sk.flatten(), ub_phi.flatten(), ub_theta.flatten(), 5, 5, 5, 5))

    ga_oscillators_final, lsq_cov = curve_fit(func_lsq_R, xdata=v, p0=ga_oscillators_initial, ydata=r, bounds=(lb, ub),
                                              maxfev=10000)

    rfit_lsq, n1_lsq, k1_lsq, n2_lsq, k2_lsq = calculate_rnk(ga_oscillators_final, v, viewing_angle)

    N = int((len(ga_oscillators_final) - 1) / 3)

    nu = ga_oscillators_final[0:N]
    gamm = ga_oscillators_final[N:N + N]
    fourpr = ga_oscillators_final[N + N:N + N + N]
    epsil = ga_oscillators_final[N + N + N]

    fig, ax = plt.subplots(6, 1, figsize=(5, 15))
    ax[0].plot(v, n1_ga, '.', 'g', label='n1_ga')
    ax[0].plot(v, n1_lsq, '.', 'b', label='n1_ga+lsq')
    ax[0].legend(bbox_to_anchor=(1, 1))
    ax[0].set_title('GA+LSQ modelled n1')

    ax[1].plot(v, n2_ga, '.', 'g', label='n2_ga')
    ax[1].plot(v, n2_lsq, '.', 'b', label='n2_ga+lsq')
    ax[1].legend(bbox_to_anchor=(1, 1))
    ax[1].set_title('GA+LSQ modelled n2')

    ax[2].plot(v, k1_ga, '.', 'g', label='k_ga')
    ax[2].plot(v, k1_lsq, '.', 'b', label='k_ga+lsq')
    ax[2].legend(bbox_to_anchor=(1, 1))
    ax[2].set_title('GA+LSQ modelled k1')

    ax[3].plot(v, k2_ga, '.', 'g', label='k2_ga')
    ax[3].plot(v, k2_lsq, '.', 'b', label='k2_ga+lsq')
    ax[3].legend(bbox_to_anchor=(1, 1))
    ax[3].set_title('GA+LSQ modelled k2')

    ax[2].plot(v, rfit_lsq, '.', 'b', label='R_ga+lsq')
    ax[2].plot(v, rfit_ga, '.', 'b', label='R_ga')
    ax[2].plot(v, r, '.', 'k', label='R')
    ax[2].legend(bbox_to_anchor=(1, 1))
    ax[2].set_title('GA+LSQ modelled R')

    ax[3].plot(v, r - rfit_lsq,'.', label='residual R')
    ax[3].set_ylim(-0.05, +0.05)
    plt.legend()
    plt.show()

    ga_lsq_chisq = {}

    chisq_val = chisq(r[1:], rfit_lsq[1:])

    ga_lsq_chisq["overall_chisq"] = chisq(r[1:], rfit_lsq[1:])

    print("Chi-square goodness of fitness is ", chisq_val)

    #     ga_lsq_chisq["chisq_val_R1"] = chisq(R_final[spectra_range['R1']], R_smoothed[spectra_range['R1']])
    #     ga_lsq_chisq["chisq_val_R2"] = chisq(R_final[spectra_range['R2']], R_smoothed[spectra_range['R2']])
    #     ga_lsq_chisq["chisq_val_R3"] = chisq(R_final[spectra_range['R3']], R_smoothed[spectra_range['R3']])
    #     ga_lsq_chisq["chisq_val_R4"] = chisq(R_final[spectra_range['R4']], R_smoothed[spectra_range['R4']])
    #     ga_lsq_chisq["chisq_val_R5"] = chisq(R_final[spectra_range['R5']], R_smoothed[spectra_range['R5']])
    #     ga_lsq_chisq["chisq_val_R6"] = chisq(R_final[spectra_range['R6']], R_smoothed[spectra_range['R6']])
    #     ga_lsq_chisq["chisq_val_R7"] = chisq(R_final[spectra_range['R7']], R_smoothed[spectra_range['R7']])

    #     print("Chi-square goodness of R1 fitness is ", chisq_val_R1)
    #     print("Chi-square goodness of R2 fitness is ", chisq_val_R2)
    #     print("Chi-square goodness of R3 fitness is ", chisq_val_R3)
    #     print("Chi-square goodness of R4 fitness is ", chisq_val_R4)
    #     print("Chi-square goodness of R5 fitness is ", chisq_val_R5)
    #     print("Chi-square goodness of R6 fitness is ", chisq_val_R6)
    #     print("Chi-square goodness of R7 fitness is ", chisq_val_R7)

    fig, ax2 = plt.subplots(1, 2, figsize=(10, 3))

    ax2[0].plot(v, r)
    for i in ga_solution[:num_oscillators]:
        ax2[0].scatter(i, r[np.abs(v - i).argmin()], label=round(i, 3), color='r')
    ax2[0].legend()
    ax2[0].set_title("GA oscillators")

    ax2[1].plot(v, r)
    for i in ga_oscillators_final[:num_oscillators]:
        ax2[1].scatter(i, r[np.abs(v - i).argmin()], label=round(i, 3), color='r')
    ax2[1].legend()
    ax2[1].set_title("GA+LSQ oscillators")
    plt.show()

    R_residuals = r - rfit_lsq

    #     fname = lf+"ga_lsq_run"+str(run)+"_Noscillators"+str(num_oscillators)
    #     #save oscillator parameters
    #     save_oscillator_parameters(fname, ga_oscillators_final, ga_lsq_chisq)
    #     #save optical constants to excel
    #     df = pd.DataFrame(np.stack((v, R_smoothed, R_final, R_residuals, n_final, k_final), axis=1), columns= ["wavenumber", "R_measured", "R_modelled", "R_residuals", "n" , "k"] )
    #     df.to_excel(fname+"_optical_constants.xlsx")

    run += 1

    if (N >= total_num_oscillators) or (run >= total_num_oscillators):
        break




