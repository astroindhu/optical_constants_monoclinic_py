from dispersion_fresnel_modules import dispersion_model_ac, fresnel_ac, calculate_rnk
from input_data import p, viewing_angle, r, v
import numpy as np
import tqdm
import pygad

def fitness_func(solution, solution_idx):
    rfit, n1, k1, n2, k2 = calculate_rnk(solution, p, viewing_angle)
    fitness = (1 - chisq(rfit, r)) * 100  # calculating euclidean distance
    return fitness


def chisq(modelled, measured):
    return (np.round(np.sum(np.square(modelled - measured) / measured), 3))


def ga_optical_constants(gene_space):
    def on_generation_progress(ga):
        #         print("Generation", ga.generations_completed)
        #         print("Generation shape", ga.population.shape)
        #         print(ga.population)
        pbar.update(1)

    def parent_selection_func(fitness, num_parents, ga_instance):
        #         print("Generation", ga_instance.generations_completed)
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        #         print("fitness sorted", fitness_sorted)

        oscillators_num = int((ga_instance.population.shape[1] - 4) / 5)

        parents = np.empty((num_parents, ga_instance.population.shape[1]))

        parent_num = 0

        for fs in fitness_sorted:
            osci_diff = np.ediff1d(ga_instance.population[fs, :oscillators_num])

            if parent_num < num_parents:
                if np.all(np.abs(osci_diff) > 10):
                    parents[parent_num, :] = ga_instance.population[fs, :].copy()
                    parent_num += 1
            else:
                break

        return parents, fitness_sorted[:num_parents]

    # num_generations = 10000
    # sol_per_pop = 100
    # num_parents_mating = 10
    #
    # num_generations = 1000
    # sol_per_pop = 50
    # num_parents_mating = 10
    num_generations = 10
    sol_per_pop = 5
    num_parents_mating = 2
    crossover_probability = 0.15
    mutation_percent_genes = (3, 2)

    with tqdm.tqdm(total=num_generations) as pbar:
        ga_instance = pygad.GA(fitness_func=fitness_func,
                               num_generations=num_generations,
                               sol_per_pop=sol_per_pop,
                               num_genes=len(gene_space),
                               num_parents_mating=num_parents_mating,
                               gene_space=gene_space,
                               gene_type=float,
                               #                                parent_selection_type='sss',
                               parent_selection_type=parent_selection_func,
                               crossover_type='two_points',
                               crossover_probability=crossover_probability,
                               mutation_type='adaptive',
                               mutation_percent_genes=mutation_percent_genes,
                               save_best_solutions=True,
                               save_solutions=True,
                               suppress_warnings=True,
                               allow_duplicate_genes=False,
                               stop_criteria="saturate_20",
                               on_generation=on_generation_progress)

        ga_instance.run()

    return ga_instance


def func_lsq_R(v, *coef):
    # disp_model_wrap: wrapper for dispersion model

    coef = np.array(coef)

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

    rfit = fresnel_ac(e_xx, e_xy, e_yy, e_zz, viewing_angle)

    return rfit
