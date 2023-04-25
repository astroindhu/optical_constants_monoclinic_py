import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

path = "/Users/astroindhu/SBU/1_Optical_Constants/optical_constants_py/MIR_optical_constants_py/monoclinic_py/monoclinic_py/monoclinic_ac/lab_ac/"

v1, r_omega1 = np.loadtxt(path+"Orth 0deg Wed Nov 09 12-36-20 2011 (GMT-05-00).CSV", unpack=True)
v2, r_omega2 = np.loadtxt(path+"Orth 45 Wed Nov 09 12-47-58 2011 (GMT-05-00).CSV", unpack=True)
v3, r_omega3 = np.loadtxt(path+"Orth 90 Wed Nov 09 13-08-04 2011 (GMT-05-00).CSV", unpack=True)
v4, r_omega4 = np.loadtxt(path+"Orth 135 Wed Nov 09 13-19-36 2011 (GMT-05-00).CSV", unpack=True)
v5, rb = np.loadtxt(path+"bkgd sample holder Mon Apr 18 15-07-09 2011 (GMT-04-00).CSV", unpack=True)
viewing_angle = 30

# R=data[:,1]/100 # convert to reflectance from % reflectance

# Clip reflectance data to desired wavenumber range.
r_omega1 = r_omega1[(v1>400) & (v1<2000)]
r_omega2 = r_omega2[(v2>400) & (v2<2000)]
r_omega3 = r_omega3[(v3>400) & (v3<2000)]
r_omega4 = r_omega4[(v4>400) & (v4<2000)]
rb = rb[(v5>400) & (v5<2000)]

v1 = v1[(v1>400) & (v1<2000)]
v2 = v2[(v2>400) & (v2<2000)]
v3 = v3[(v3>400) & (v3<2000)]
v4 = v4[(v4>400) & (v4<2000)]
v5 = v5[(v5>400) & (v5<2000)]

# Subtract background spectrum of empty chamber with sample holder. Comment out the next line to skip this step.
r_omega1=r_omega1-rb
r_omega2=r_omega2-rb
r_omega3=r_omega3-rb
r_omega4=r_omega4-rb

#Change wavenumber vector direction for each reflectance spectrum and combine into one array.
v = np.hstack((v1,v2,v3,v4))

#Polarization angle with respect to the a-axis of the crystal for each reflectance spectrum.
omega_1=0
omega_2=45
omega_3=90
omega_4=135
#extend omega into a vector with length of each reflectance spectrum and combine into one array
omega_1 = np.full(len(v1),omega_1)
omega_2 = np.full(len(v2),omega_2)
omega_3 = np.full(len(v3),omega_3)
omega_4 = np.full(len(v4),omega_4)
omega=np.hstack((omega_1,omega_2,omega_3,omega_4))

# create xdata array for fitting the reflectance spectrum r=fn(v,omega)

p1 = np.column_stack((v1,omega_1))
p2 = np.column_stack((v2,omega_2))
p3 = np.column_stack((v3,omega_3))
p4 = np.column_stack((v4,omega_4))
p = np.column_stack((v,omega))

# convert to reflectance from % reflectance
r_omega1 = 0.01*r_omega1
r_omega2 = 0.01*r_omega2
r_omega3 = 0.01*r_omega3
r_omega4 = 0.01*r_omega4
r = np.hstack((r_omega1, r_omega2, r_omega3, r_omega4))


plt.plot(v1, r_omega1, label="0 deg")
plt.plot(v2, r_omega2, label="45 deg")
plt.plot(v3, r_omega3, label="90 deg")
plt.plot(v4, r_omega4, label="135 deg")
# plt.plot(v5, rb, label="background sample?")
plt.xlabel('wavenumber (cm${^{-1}}$)')
plt.ylabel("Reflectance")
plt.legend()
plt.tight_layout()
plt.show()