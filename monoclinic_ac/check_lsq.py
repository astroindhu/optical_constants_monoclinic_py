import numpy as np
from input_data import r, p, v, v1, r_omega1, r_omega2, r_omega3, r_omega4, viewing_angle
from scipy.optimize import curve_fit
from genetic_algorithm_modules import func_lsq_R
from dispersion_fresnel_modules import calculate_rnk
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

coefg = np.array([1128.539222,1108.826422,1020.322886,1034.233524,1012.274532,990.006636,744.45369,720.778782,638.158624,579.071147,570.831658,563.501451,417.02516,1135.343215,1102.849205,1020.661924,992.40544,774.47989,752.369921,725.759822,610.550125,543.069816,29.536676,29.221712,41.347021,24.006984,35.892791,28.258788,54.175377,29.799001,17.933558,24.735389,11.552889,8.822451,33.293212,39.831837,42.528459,42.663048,32.121415,22.329749,23.549608,25.539595,17.510378,16.949685,54483.01613,212239.2772,363391.8915,141988.0592,172764.6226,150277.6745,15444.9342,20719.67039,29321.22705,42621.51927,74252.33067,35780.59196,115495.0724,14854.15997,32799.52997,259002.78,387229.13,15639.11,15823.77,18424.89999,11599.097,31705.61,120.4491,67.665757,90.466159,-3.755058,179.179185,149.034945,45.776972,-50.061972,120.892857,57.615464,47.644193,51.495434,-57.40553,0,0,0,0,0,0,0,0,0,90,90,90,90,90,90,90,90,90,90,90,90,90,0,0,0,0,0,0,0,0,0,1.755951,0.004,2.245888,2.787198])
lb = np.array([900,900,900,900,900,800,500,500,500,500,500,400,400,1135,1102,1020,992,774,752,725,610,543,0,0,0,0,0,0,0,0,0,0,0,0,0,39,42,42,32,22,23,25,17,16,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,14854,32799,259002,387229,15639,15823,18424,11599,31705,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-1,-1,-1,-1,-1,-1,-1,-1,-1,89,89,89,89,89,89,89,89,89,89,89,89,89,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,0.004,1,0.5])
ub = np.array([1250,1250,1250,1250,1100,1100,1000,1000,1000,1000,700,700,500,1135.4,1102.9,1020.7,992.5,774.5,752.4,725.8,610.6,543.1,100,100,100,100,100,100,100,100,100,100,100,100,100,39.9,42.6,42.8,32.3,22.4,23.6,25.6,17.6,17,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,1000000,14854.2,32799.6,259002.8,387229.2,15639.2,15823.8,18425,11599.1,31705.7,180,180,180,180,180,180,180,180,180,180,180,180,180,1,1,1,1,1,1,1,1,1,91,91,91,91,91,91,91,91,91,91,91,91,91,1,1,1,1,1,1,1,1,1,5,1,5,5])

N = int((len(coefg) - 4) / 5)

plt.plot(v1,r_omega1, label="lab")
for i in coefg[:N]:
    plt.scatter(i, r_omega1[np.abs(v1 - i).argmin()], color='r')
plt.title("oscillators")
plt.tight_layout()
plt.show()

print("started non linear least square fitting")
ga_oscillators_final, lsq_cov = curve_fit(func_lsq_R, xdata=v, p0=coefg, ydata=r, bounds=(lb, ub),
                                          maxfev=10000)
print("completed non linear least square fitting")

print("Now calculating r,n,k values")
rfit_lsq, n1_lsq, k1_lsq, n2_lsq, k2_lsq = calculate_rnk(ga_oscillators_final, p, viewing_angle)
print("Completed calculating r,n,k values")

N = int((len(ga_oscillators_final) - 4) / 5)

plt.plot(v1,r_omega1, label="lab")
for i in ga_oscillators_final[:N]:
    plt.scatter(i, r_omega1[np.abs(v1 - i).argmin()], color='r')
plt.title("oscillators")
plt.tight_layout()
plt.show()

L = len(v1)

rfit_lsq_omega1 = rfit_lsq[:L]
rfit_lsq_omega2 = rfit_lsq[L:2 * L]
rfit_lsq_omega3 = rfit_lsq[2 * L:3 * L]
rfit_lsq_omega4 = rfit_lsq[3 * L:4 * L]

fig, ax = plt.subplots(4, 1, figsize=(17, 20))

ax[0].plot(v1, rfit_lsq_omega1, 'b', label='R_modelled')
ax[0].plot(v1, r_omega1, 'k', label='R')
ax[0].set_xlim(2000, 400)
ax[0].legend(bbox_to_anchor=(1, 1))
ax[0].set_title(' modelled R 0 deg')

ax[1].plot(v1, rfit_lsq_omega2, 'b', label='R_modelled')
ax[1].plot(v1, r_omega2, 'k', label='R')
ax[1].set_xlim(2000, 400)
ax[1].legend(bbox_to_anchor=(1, 1))
ax[1].set_title(' modelled R 45 deg')

ax[2].plot(v1, rfit_lsq_omega3, 'b', label='R_modelled')
ax[2].plot(v1, r_omega3, 'k', label='R')
ax[2].set_xlim(2000, 400)
ax[2].legend(bbox_to_anchor=(1, 1))
ax[2].set_title(' modelled R 90 deg')

ax[3].plot(v1, rfit_lsq_omega4, 'b', label='R_modelled')
ax[3].plot(v1, r_omega4, 'k', label='R')
ax[3].set_xlim(2000, 400)
ax[3].legend(bbox_to_anchor=(1, 1))
ax[3].set_title('modelled R 135 deg')
plt.legend()
plt.tight_layout()
plt.show()

n1_lsq = n1_lsq[:L]
n2_lsq = n2_lsq[:L]
k1_lsq = k1_lsq[:L]
k2_lsq = k2_lsq[:L]

fig, ax = plt.subplots(2, 1, figsize=(8, 10))
ax[0].plot(v1, n1_lsq, 'b', label='n1')
ax[0].plot(v1, n2_lsq, 'k', label='n2')
ax[0].set_xlim(2000, 400)
ax[0].legend(bbox_to_anchor=(1, 1))

ax[1].plot(v1, k1_lsq, 'b', label='k1')
ax[1].plot(v1, k2_lsq, 'k', label='k2')
ax[1].set_xlim(2000, 400)
ax[1].legend(bbox_to_anchor=(1, 1))
ax[1].set_yscale('log')
plt.legend()
plt.show()



import pickle
with open('check_lsq_results.pkl', 'wb') as f:
    pickle.dump([v, r, rfit_lsq, n1_lsq, k1_lsq, n2_lsq, k2_lsq], f)