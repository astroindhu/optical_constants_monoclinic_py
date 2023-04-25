# fresnel_ac: fresnel model for non-normal incidence
# function 2: dispersion_model
# function 3: calculate r, n, k using fresnel and dispersion
import numpy as np

def sind(deg):
    return np.sin(deg * (np.pi / 180))


def cosd(deg):
    return np.cos(deg * (np.pi / 180))


# fresnel: fresnel model for non-normal incidence
def fresnel_ac(e_xx, e_xy, e_yy, e_zz, alpha):
    # alpha = viewing_angledeg
    # % Fresnel reflectance model for alpha degree incidence
    # %
    # % inputs: e_xx,e_xy,e_yy,e_zz - dielectric tensor
    # %
    # % returs: R - reflectance

    k_y = sind(alpha)

    # make e's be 1x1xv
    e_xx = np.reshape(e_xx, (1, 1, np.shape(e_xx)[0]))
    e_xy = np.reshape(e_xy, (1, 1, np.shape(e_xy)[0]))
    e_yy = np.reshape(e_yy, (1, 1, np.shape(e_yy)[0]))
    e_zz = np.reshape(e_zz, (1, 1, np.shape(e_zz)[0]))

    # calculate M matrix
    k1 = -e_zz * (e_xx + e_yy) + np.square(k_y) * (e_yy + e_zz)
    k2 = -4 * (np.square(e_xy) + e_yy * (np.square(k_y) - e_xx)) * (np.square(k_y) - e_zz) * e_zz

    gamm1 = np.sqrt((-1 / (2 * e_zz)) * (k1 + np.sqrt(np.square(k1) + k2)))
    gamm3 = np.sqrt((1 / (2 * e_zz)) * (-k1 + np.sqrt(np.square(k1) + k2)))

    d_0_inv_2d = np.array([[0.5, 0.5 / (cosd(alpha)), 0, 0],
                           [0.5, -0.5 / (cosd(alpha)), 0, 0],
                           [0, 0, 0.5 / (cosd(alpha)), 0.5],
                           [0, 0, -0.5 / (cosd(alpha)), 0.5]])
    d_0_inv = np.repeat(d_0_inv_2d[:, :, np.newaxis], np.shape(e_xx)[2], axis=2)
    zeds = np.zeros((1, 1, np.shape(e_xx)[2]))

    d_1 = np.concatenate(
        (np.concatenate((e_xy*(1-np.divide(np.square(k_y),e_zz)), zeds, e_xy * (1 - np.square(k_y) / e_zz), zeds), axis=1),
         np.concatenate(
             (gamm1 * e_xy * (1 - np.square(k_y) / e_zz), zeds, gamm3 * e_xy * (1 - np.square(k_y) / e_zz), zeds),
             axis=1),
         np.concatenate(((1 - np.square(k_y) / e_zz) * (np.square(gamm1) - (e_xx - np.square(k_y))), zeds,
                         (1 - np.square(k_y) / e_zz) * (np.square(gamm3) - (e_xx - np.square(k_y))), zeds), axis=1),
         np.concatenate((np.power(gamm1, 3) - (gamm1 * (e_xx - np.square(k_y))), zeds,
                         np.power(gamm3, 3) - (gamm3 * (e_xx - np.square(k_y))), zeds), axis=1)), axis=0)

    M = np.empty([4, 4, np.shape(d_1)[2]], dtype="complex")

    for i in np.arange(np.shape(d_1)[2]):
        M[:, :, i] = np.matmul(d_0_inv[:, :, i], d_1[:, :, i])

    # calculate reflectance coefficients
    denom = (M[0, 0, :] * M[2, 2, :]) - (M[0, 2, :] * M[2, 0, :])
    r_xx = np.divide((M[1, 0, :] * M[2, 2, :]) - (M[1, 2, :] * M[2, 0, :]), denom)
    # r_xx = np.reshape(r_xx, (np.shape(r_xx)[2], 1))
    r_xy = np.divide((M[3, 0, :] * M[2, 2, :]) - (M[3, 2, :] * M[2, 0, :]), denom)
    # r_xy = np.reshape(r_xy, (np.shape(r_xy)[2], 1))

    # calculate reflectance
    R = np.square(np.abs(r_xx)) + np.square(np.abs(r_xy))

    return R


def dispersion_model_ac(nu, gamm, Sk, phi, theta, epsilxx, epsilxy, epsilyy, epsilzz, p):
    # dispersion_model: Mayerhofer et al.
    #
    # inputs: p - predictors (frequency list, omega)
    #         gamm - damping parameter
    #         Sk - bandwidth parameter
    #         epsil - bulk dielectric constants
    #
    # returns: n1,n2,k1,k2 - optical constants
    #          e_xx,e_xy,e_yy,e_zz - dielectric tensor

    v = p[:, 0]
    omega = p[:, 1]

    v = v.reshape(-1,1) # make v be Mx1
    omega = omega.reshape(-1,1)  #make omega be Mx1

    # make nu, Sk, gamm be 1xN
    nu = nu[:, None].T
    Sk = Sk[:, None].T
    gamm = gamm[:, None].T
    phi = phi[:, None].T
    theta = theta[:, None].T

    v = np.tile(v, (1, np.shape(nu)[1]))
    omega = np.tile(omega, (1, np.shape(nu)[1]))
    phi = np.tile(phi, (len(v), 1))
    theta = np.tile(theta, (len(v), 1))
    nu = np.tile(nu, (len(v), 1))
    Sk = np.tile(Sk, (len(v), 1))
    gamm = np.tile(gamm, (len(v), 1))

    # calculate dielectric tensor

    uxx = np.square(sind(theta)) * np.square(cosd(phi))
    uxy = np.square(sind(theta)) * cosd(phi) * sind(phi)
    uyy = np.square(sind(theta)) * np.square(sind(phi))
    uzz = np.square(cosd(theta))

    denom = np.square(nu) - np.square(v) - (1j * gamm * v)

    L_k_xx = np.divide((uxx * Sk), denom)
    L_k_xy = np.divide((uxy * Sk), denom)
    L_k_yy = np.divide((uyy * Sk), denom)
    L_k_zz = np.divide((uzz * Sk), denom)

    e_11 = epsilxx + np.sum(L_k_xx, axis=1)
    e_12 = epsilxy + np.sum(L_k_xy, axis=1)
    e_22 = epsilyy + np.sum(L_k_yy, axis=1)
    e_33 = epsilzz + np.sum(L_k_zz, axis=1)

    # rotate dielectric tensor to match coordinate system
    e_xx = (e_11 * np.square(cosd(omega[:,0]))) + (e_22 * np.square(sind(omega[:,0]))) + (2 * e_12 * cosd(omega[:,0]) * sind(omega[:,0]))
    e_xy = (e_12 * np.square(cosd(omega[:,0]))) - (e_12 * np.square(sind(omega[:,0]))) + (
            (e_22 - e_11) * cosd(omega[:,0]) * sind(omega[:,0]))
    e_yy = (e_22 * np.square(cosd(omega[:,0]))) + (e_11 * np.square(sind(omega[:,0]))) + (2 * e_12 * cosd(omega[:,0]) * sind(omega[:,0]))
    e_zz = e_33

    # calculate optical constants
    m1sq = np.zeros((len(v), 1), dtype="complex")
    m2sq = np.zeros((len(v), 1), dtype="complex")
    m1sq[0] = (e_xx[0] + e_yy[0])/2 + np.sqrt((np.square(e_xx[0] - e_yy[0]))/4 + np.square(e_xy[0]))
    m2sq[0] = (e_xx[0] + e_yy[0])/2 - np.sqrt((np.square(e_xx[0] - e_yy[0]))/4 + np.square(e_xy[0]))

    # enforce continuity
    for i in np.arange(1, len(v)):
        sl1 = (e_xx[i] + e_yy[i])/2 + np.sqrt((np.square(e_xx[i] - e_yy[i]))/4 + np.square(e_xy[i]))
        sl2 = (e_xx[i] + e_yy[i])/2 - np.sqrt((np.square(e_xx[i] - e_yy[i]))/4 + np.square(e_xy[i]))

        if np.abs(sl1 - m1sq[i - 1]) < (np.abs(sl1 - m2sq[i - 1])):
            m1sq[i] = sl1
            m2sq[i] = sl2
        else:
            m1sq[i] = sl2
            m2sq[i] = sl1

    m1 = np.sqrt(m1sq)
    m2 = np.sqrt(m2sq)
    n1 = np.real(m1)
    k1 = np.imag(m1)
    n2 = np.real(m2)
    k2 = np.imag(m2)

    return n1, n2, k1, k2, e_xx, e_xy, e_yy, e_zz


def calculate_rnk(coef, p, alpha):
    # disp_model_wrap_ac: wrapper for dispersion model.
    #    lsqcurvefit requires all coefficients in one vector or rectangular
    #    matrix. This function separates the vector back into oscillator
    #    parameters to more readily input into the actual dispersion model.
    #
    # inputs: p - predictors (frequency list, omega)
    #         r - measured reflectance
    #
    # returns: rfit - reflectance fit
    #          n1,n2,k1,k2 - optical constants

    N = int((len(coef) - 4) / 5)
    #     if isinstance(N, int):
    #         print('ERROR: disp_model_wrap_sbu: bad length for coefficient list!')

    nu = coef[0:N]
    gamm = coef[N:2 * N]
    Sk = coef[2 * N:3 * N]
    phi = coef[3 * N:4 * N]
    theta = coef[4 * N:5 * N]
    epsilxx = coef[5 * N]
    epsilxy = coef[(5 * N) + 1]
    epsilyy = coef[(5 * N) + 2]
    epsilzz = coef[(5 * N) + 3]

    n1, n2, k1, k2, e_xx, e_xy, e_yy, e_zz = dispersion_model_ac(nu, gamm, Sk, phi, theta, epsilxx, epsilxy, epsilyy,
                                                                 epsilzz, p)

    rfit = fresnel_ac(e_xx, e_xy, e_yy, e_zz, alpha)

    return rfit, n1, k1, n2, k2
