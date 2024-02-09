import numpy as np
import sympy as sp

from aux import fBose, PSD

def decompose_spe(spe, w_sp, sp_para_dict, para_dict, condition_dict, npsd, pade=1):
    if (sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[1] == 0):
        imag_part = sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[0]
    else:
        imag_part = sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)

    poles = sp.nroots(denom_get_para)
    float(sp.re(poles[0]))

    expn = []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    etal = []
    etar = []
    etaa = []

    expn = np.array(expn)

    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]

    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    expn = list(expn[expn_imag_sort])
    pole, resi = PSD(npsd, 1, pade)
    beta = para_dict['beta']
    temp = 1 / beta

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}) *
                     fBose(-1.J * expn_val_cc[ii] / temp, pole, resi))))

        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}) *
                     fBose(-1.J * expn_val_cc[ii + 1] / temp, pole, resi))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_n_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_n_cc[ii]}) *
                     fBose(-1.J * expn_val_n_cc[ii] / temp, pole, resi))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    f = numer_get_para / np.multiply.reduce(w_sp - poles_allplane)
    f = sp.lambdify(w_sp, f)

    for inma in range(len(pole)):
        zomg = -1.J * pole[inma] * temp
        jsum = np.sum(f(zomg))
        expn.append(pole[inma] * temp)
        etal.append(-2.J * resi[inma] * temp * jsum)
        etar.append(np.conj(etal[-1]))
        etaa.append(np.abs(etal[-1]))

    etal = np.array(etal)
    etar = np.array(etar)
    etaa = np.array(etaa)
    expn = np.array(expn)
    return etal, etar, etaa, expn

if __name__ == '__main__':
    eta = 1
    gam = 1
    temp = 1
    beta = 1 / temp
    npsd = 2

    w_sp, eta_sp, gamma_sp, beta_sp = sp.symbols(
        r"\omega, \eta, \gamma, \beta", real=True)

    phixx_sp = 2 * eta_sp * gamma_sp / (gamma_sp - sp.I * w_sp)
    spe_vib_sp = phixx_sp
    sp_para_dict = {eta_sp: eta, gamma_sp: gam}
    condition_dict = {}
    para_dict = {'beta': beta}
    etal, etar, etaa, expn = decompose_spe(spe_vib_sp, w_sp, sp_para_dict, para_dict,
                                           condition_dict, npsd)
    
    spe_out = np.vstack((etal, etar))
    spe_out = np.vstack((spe_out, etaa))
    spe_out = np.vstack((spe_out, expn))
    print(spe_out)
    np.savetxt(fname="D:\my work\DQME_modes\spe_out.dat", X=spe_out)