import pandas as pd

def save_oscillator_parameters(fname, coef, chisq_dict):
    oscillator_parameters_df_dict = {}

    N = int((len(coef) - 4) / 5)

    nu = coef[0:N]
    gamm = coef[N:2 * N]
    Sk = coef[2 * N:3 * N]
    phi = coef[3 * N:4 * N]
    theta = coef[4 * N:5 * N]
    epsilxx = coef[5 * N]
    epsilxy = coef[(5 * N) + 1]
    epsilyy = coef[(5 * N) + 2]
    epsilzz = coef[(5 * N) + 3]

    oscillator_parameters_df_dict = {'nu%s' % 1: nu[0]}
    oscillator_parameters_df_dict.update({'gamm%s' % 1: gamm[0]})
    oscillator_parameters_df_dict.update({'Sk%s' % 1: Sk[0]})
    oscillator_parameters_df_dict.update({'phi%s' % 1: phi[0]})
    oscillator_parameters_df_dict.update({'theta%s' % 1: theta[0]})
    oscillator_parameters_df_dict.update({'epsilxx': epsilxx})
    oscillator_parameters_df_dict.update({'epsilxy': epsilxy})
    oscillator_parameters_df_dict.update({'epsilyy': epsilyy})
    oscillator_parameters_df_dict.update({'epsilzz': epsilzz})

    for i in range(1, N + 1):
        oscillator_parameters_df_dict = {'nu%s' % i: nu[i - 1]}
        oscillator_parameters_df_dict.update({'gamm%s' % i: gamm[i - 1]})
        oscillator_parameters_df_dict.update({'Sk%s' % i: Sk[i - 1]})
        oscillator_parameters_df_dict.update({'phi%s' % i: phi[i - 1]})
        oscillator_parameters_df_dict.update({'theta%s' % i: theta[i - 1]})

    for k, v in chisq_dict.items():
        oscillator_parameters_df_dict.update({k: v})

    df = pd.DataFrame.from_dict(oscillator_parameters_df_dict, orient='index')
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_index()
    df.to_excel(fname + "_oscillator_params.xlsx")
