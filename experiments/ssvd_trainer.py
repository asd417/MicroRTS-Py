# Dependencies required are
import os
import glob
import math
import tqdm
import torch # with cuda
import torch.jit
import torch.nn as nn
import statistics

import numpy as np
import gymnasium as gym
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env_custom import MicroRTSGridModeVecEnv
from gym_microrts.envs.vec_mcts_env import MicroRTSMCTSEnv

from torch.utils.tensorboard import SummaryWriter
from experiments.ssvd.ssvd import SSVDVariable, SSVDModel
#from experiments.commons import load_or_create_pop, save_pop, write_log, apply_fitness
import experiments.commons
import pyade.commons

def crossover_simple(parent1, parent2):
    """Performs single-point crossover between two matrices."""
    rows, _ = parent1.shape
    crossover_pointr = torch.randint(0, rows, (1,),device=parent1.device).item()  # Choose a row to swap from
    child = torch.clone(parent1)
    # Create a child matrix by swapping rows below crossover point
    child[crossover_pointr:, :] = parent2[crossover_pointr:, :]  # Swap lower part
    return child

def _crossover_matrix(parent1 : torch.Tensor, parent2 : torch.Tensor):
    assert str(parent1.shape) == str(parent2.shape), f"Can not crossover two matrices with different shapes: {parent1.shape} {parent2.shape}"
    #assert not (parent1.shape[0] == 1 and parent2.shape[0] == 1), f"Can not crossover two matrices with size 1x1"
    #print(f"shape: {parent1.shape}")
    # example of crossing over two matrices with shape (3,2)

    # maxsel = 3 - 1 + 2 - 1 = 3
    # places: 1 2 3 (because crossover at position 0 does nothing)
    
    # randint = 3
    # 3 > (3-1) -> crossover col
    # 3 - (3-1) = 3 - 2 = 1
    # crossover col at position 1

    # if randint = 2
    # 2 <= (3-1) -> crossover row
    # crossover col at position 2
    maxsel = parent1.shape[0] - 1 + parent1.shape[1] - 1
    random_int = torch.randint(1, maxsel+1, (1,))
    if random_int > parent1.shape[0]-1: # crossover col
        random_int -= (parent1.shape[0]-1)
        child = torch.cat((parent1[:, :random_int], parent2[:, random_int:]), dim=1)
    else:
         # crossover row
        child = torch.cat((parent1[:random_int, :], parent2[random_int:, :]), dim=0)
    return child

def crossover_gam(parent1, parent2, ssvd : SSVDVariable):
    """Performs single-point crossover between two matrices."""
    p1w1s, p1w2s, p1wO = ssvd.chromosome_to_weights(parent1)
    p2w1s, p2w2s, p2wO = ssvd.chromosome_to_weights(parent2)
    pow1s = torch.cat([_crossover_matrix(p1w1, p2w1) for p1w1, p2w1 in zip(p1w1s, p2w1s)]).flatten()
    pow2s = torch.cat([_crossover_matrix(p1w2, p2w2) for p1w2, p2w2 in zip(p1w2s, p2w2s)]).flatten()
    powO = _crossover_matrix(p1wO,p2wO).flatten()
    print(f"{pow1s.shape}, {pow2s.shape}, {powO.shape}")
    return torch.reshape(torch.cat((pow1s, pow2s, powO)).flatten(),(-1,1))

def mutate_multivariate_gaussian(matrix, mutation_rate=0.1):
    """
    Applies mutation by adding noise sampled from a multivariate Gaussian distribution.
    The mean is zero, and the covariance matrix is an identity matrix scaled by a factor.
    """
    
    #print(f"mutate_multivariate_gaussian {matrix.shape}")
    rows, cols = matrix.shape
    mutation_mask = torch.rand(rows, cols, device=matrix.device) < mutation_rate  # Decide mutation points
    
    # Create a covariance matrix (identity for simplicity)
    cov_matrix = torch.eye(cols, device=matrix.device) * 0.1  # Scaling factor

    # Sample mutation noise using multivariate Gaussian
    mean = torch.zeros(cols, device=matrix.device)  # Mean is zero
    mvn_noise = torch.distributions.MultivariateNormal(mean, cov_matrix)
    
    # Apply the mutation only where the mask is True
    for i in range(rows):
        if mutation_mask[i].any():  # If any column in row i is mutated
            noise_sample = mvn_noise.sample()  # Sample once per row
            matrix[i, mutation_mask[i]] += noise_sample[mutation_mask[i]]  # Apply only where mask is True
            
    return matrix

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

#TODO let this spawn multiple environments at the same time. The microrts client already supports it.
#need to revert some of the work I did on the java side because I might have broken it.
def fitness(envs, chromosome, ssvd, maxstep = 3000, render=False, record=False):
    weights1, weights2, weightsO = ssvd.chromosome_to_weights(chromosome)
    obs = envs.reset()
    prev_r = None
    prev_d = None
    scores = np.zeros((envs.num_envs,))
    dones = np.ones((envs.num_envs,))
    for i in range(maxstep):
        if render:
            if record:
                envs.render(mode="rgb_array")
            else:
                envs.render()
        if not render and record:
            envs.render(mode="rgb_array")
        model = SSVDModel(envs, ssvd.k, device=weights1.device) # full k
        action = model(obs, weights1, weights2, weightsO)
        obs, reward, done, info = envs.step(action.to("cpu").detach().numpy())
        # dones is an array of booleans. The element is true if the step ends the game. after that, new game is started immediately.
        # below ensures that scores are calculated for exactly 1 game per env
        dones -= done.astype(int)
        dones = np.clip(dones, 0, None)
        scores += dones * reward # making sure to only record score from completed games only
        if prev_r != str(scores) or prev_d != str(dones):
            #print(f"{scores} \t {dones}")
            prev_r = str(scores)
            prev_d = str(dones)
        if np.all(dones == 0):
            break
    del weights1
    del weights2
    del weightsO
    return sum(scores) / float(envs.num_envs)

def fitness_mcts(envs, chromosome, ssvd, maxstep=3000, render=False, record=False):
    # chromosome is a 1D vector
    # chromosomes are turned into weight matrices on java
    chromosome = chromosome.squeeze()
    envs.reset(chromosome)
    prev_r = None
    prev_d = None
    scores = np.zeros((envs.num_envs,))
    dones = np.ones((envs.num_envs,))
    for i in range(maxstep):
        if render:
            if record:
                envs.render(mode="rgb_array")
            else:
                envs.render()
        if not render and record:
            envs.render(mode="rgb_array")
        _, reward, done, info = envs.step()
        dones -= done.astype(int)
        dones = np.clip(dones, 0, None)
        scores += dones * reward # making sure to only record score from completed games only
        if prev_r != str(scores) or prev_d != str(dones):
            #print(f"{scores} \t {dones}")
            prev_r = str(scores)
            prev_d = str(dones)
        if np.all(dones == 0):
            break
    return sum(scores) / float(envs.num_envs)


def get_logger(name, directory="runs/") -> SummaryWriter:
    log_dir = f"{directory}{name}"
    writer = SummaryWriter(log_dir)
    return writer


def run_test_de(ssvd, envs, pop_size, max_iter, device, fitness_func, render=False, record=False, maxstep=3000, logdir="", ptdir=""):
    #population = pyade.commons.init_population(population_size, individual_size, bounds)
    name = os.path.basename(logdir).split(".")[0]
    writer = get_logger(name)
    gi, p = experiments.commons.load_or_create_pop(pop_size, ssvd, logdir, ptdir)
    mutation_rate = 0.5

    win = False
    while win == False:
        if(max_iter < gi):
            win = True
            break

        progress = tqdm.tqdm(total=len(p), desc=f"{name} Generation {gi}/{max_iter}")
        win, fits, best_chromosome, best_fitness_single_gen = experiments.commons.apply_fitness(gi,envs,ssvd,p,fitness_func,maxstep,render,writer=writer,progress=progress)
        avg = sum(fits) / len(fits)
        std = statistics.stdev(fits)
        writer.add_scalars(f"{name}/Fitness", {
            "Best Fitness": best_fitness_single_gen,
            "Average Fitness": avg,
            "Standard Deviation": std,
            "Upper Bound": avg + std,
            "Lower Bound": avg - std
        }, gi)
        logstr = f"Generation {gi} {name} Highest: {best_fitness_single_gen} Average: {avg} StDev: {std}"
        tqdm.tqdm.write(logstr)
        experiments.commons.write_log(logstr, logdir)

# openai es
def run_test_es(ssvd, envs, pop_size, max_iter, device, fitness_func, render=False, record=False, maxstep=3000, logdir="", ptdir=""):
    name = os.path.basename(logdir).split(".")[0]
    sigma = 0.1    # noise standard deviation
    alpha = 0.001  # learning rate
    gen_start, w = experiments.commons.load_or_create_pop(1, ssvd, logdir, ptdir, device=device)
    writer = get_logger(name)
    
    for i in range(gen_start, max_iter):
        w = w.squeeze(0)
        progress = tqdm.tqdm(total=pop_size, desc=f"{name} Generation {i}/{max_iter}")
        N = torch.randn((pop_size, ssvd.get_chromosome_size(), 1), device=device)
        R = torch.zeros(pop_size)
        best_fitness_single_gen = 0

        for j in range(pop_size):
            w_try = w + sigma*N[j]
            f = fitness_func(envs, w_try, ssvd, maxstep=maxstep, render=render)
            R[j] = f
            writer.add_histogram("Chromosome", w_try, i)
            if f > best_fitness_single_gen:
                best_fitness_single_gen = f
            tqdm.tqdm.write(f"Fitness: {f}")
            progress.update(1)
        avg = torch.mean(R)
        std = torch.std(R)
        writer.add_scalars(f"{name}/Fitness", {
            "Best Fitness": best_fitness_single_gen,
            "Average Fitness": avg,
            "Standard Deviation": std,
            "Upper Bound": avg + std,
            "Lower Bound": avg - std
        }, i)
        logstr = f"Generation {i} {name} Highest: {best_fitness_single_gen} Average: {avg} StDev: {std}"
        tqdm.tqdm.write(logstr)
        experiments.commons.write_log(logstr, logdir)
        A = (R - torch.mean(R)) / torch.std(R)
        N = N.squeeze()
        w = w.squeeze(-1)
        #print(f"N.T @ A {(N.T @ A).shape}")
        #print(f"w {w.shape}")
        w = w + alpha/(pop_size*sigma) * N.T @ A
        w = w.unsqueeze(0).unsqueeze(-1)
        experiments.commons.save_pop(w, ptdir)

def run_test_ga(ssvd, envs, pop_size, max_iter, device, fitness_func, render=False, record=False, elitism=0.1, maxstep=3000, logdir="", ptdir=""):
    name = os.path.basename(logdir).split(".")[0]
    writer = get_logger(name)
    gi, p = experiments.commons.load_or_create_pop(pop_size, ssvd, logdir, ptdir)
    mutation_rate = 0.5

    best_chromosome = None
    best_fitness = 0
    win = False
    while win == False:
        if(max_iter < gi):
            win = True
            break
        progress = tqdm.tqdm(total=len(p), desc=f"{name} Generation {gi}/{max_iter}")
        win, fits, best_chromosome, best_fitness_single_gen= experiments.commons.apply_fitness(gi,envs,ssvd,p,fitness_func,maxstep,render,writer=writer,progress=progress)
        avg = sum(fits) / len(fits)
        std = statistics.stdev(fits)
        writer.add_scalars(f"{name}/Fitness", {
            "Best Fitness": best_fitness_single_gen,
            "Average Fitness": avg,
            "Standard Deviation": std,
            "Upper Bound": avg + std,
            "Lower Bound": avg - std
        }, gi)
        logstr = f"Generation {gi} {name} Highest: {best_fitness_single_gen} Average: {avg} StDev: {std}"
        tqdm.tqdm.write(logstr)
        experiments.commons.write_log(logstr, logdir)
        if(not win):
            ev_p = list(zip(p,fits))
            ev_p_sorted = sorted(ev_p, key=lambda x: x[1], reverse=True) # sort by fitness from highest to lowest
            elites = int(pop_size * elitism)
            ev_p_sorted = ev_p_sorted[:elites]
            new_p = []
            print("Preparing next generation...")
            for i in range(pop_size - elites): # GA
                parent1, parent2 = experiments.commons.roulette_selection(ev_p, device)
                #print("Crossover...")
                co = crossover_simple(parent1, parent2)
                #print("Mutating...")
                mutate = mutate_multivariate_gaussian(co, mutation_rate)
                #print("Done Individual Creation")
                new_p.append(mutate)
                
            p = [tup[0] for tup in ev_p_sorted] + new_p
            gi += 1
        else:
            logstr = f"Training Done | Best Fitness: {best_fitness}"
            tqdm.tqdm.write(logstr)
            experiments.commons.write_log(logstr, logdir)
            #logstr = f"Chromosome: {chromosome}"
            tqdm.tqdm.write(logstr)
            experiments.commons.write_log(logstr, logdir)
            experiments.commons.save_pop(best_chromosome, name=name+"_best")
        experiments.commons.save_pop(p, ptdir)
    envs.close()

# GA but crossover happens between weights of same shapes
def run_test_gam(ssvd, envs, pop_size, max_iter, device, fitness_func, render=False, record=False, elitism=0.1, maxstep=3000, logdir="", ptdir=""):
    name = os.path.basename(logdir).split(".")[0]
    writer = get_logger(name)
    gi, p = experiments.commons.load_or_create_pop(pop_size, ssvd, logdir, ptdir)
    mutation_rate = 0.5

    best_chromosome = None
    best_fitness = 0
    win = False
    while win == False:
        if(max_iter < gi):
            win = True
            break

        progress = tqdm.tqdm(total=len(p), desc=f"{name} Generation {gi}/{max_iter}")
        win, fits, best_chromosome,best_fitness_single_gen = experiments.commons.apply_fitness(gi,envs,ssvd,p,fitness_func,maxstep,render,writer=writer,progress=progress)
        avg = sum(fits) / len(fits)
        std = statistics.stdev(fits)
        writer.add_scalars(f"{name}/Fitness", {
            "Best Fitness": best_fitness_single_gen,
            "Average Fitness": avg,
            "Standard Deviation": std,
            "Upper Bound": avg + std,
            "Lower Bound": avg - std
        }, gi)
        logstr = f"Generation {gi} {name} Highest: {best_fitness_single_gen} Average: {avg} StDev: {std}"
        tqdm.tqdm.write(logstr)
        experiments.commons.write_log(logstr, logdir)
        if(not win):
            ev_p = list(zip(p,fits))
            ev_p_sorted = sorted(ev_p, key=lambda x: x[1], reverse=True) # sort by fitness from highest to lowest
            elites = int(pop_size * elitism)
            ev_p_sorted = ev_p_sorted[:elites]
            new_p = []
            print("Preparing next generation...")
            for i in range(pop_size - elites): # GA - Matrix
                parent1, parent2 = experiments.commons.roulette_selection(ev_p, device)
                #print("Crossover...")
                co = crossover_gam(parent1, parent2, ssvd)
                #print("Mutating...")
                mutate = mutate_multivariate_gaussian(co, mutation_rate)
                #print("Done Individual Creation")
                new_p.append(mutate)
                
            p = [tup[0] for tup in ev_p_sorted] + new_p
            gi += 1
        else:
            logstr = f"Training Done | Best Fitness: {best_fitness}"
            tqdm.tqdm.write(logstr)
            experiments.commons.write_log(logstr, logdir)
            tqdm.tqdm.write(logstr)
            experiments.commons.write_log(logstr, logdir)
            experiments.commons.save_pop(best_chromosome, name=name+"_best")
        experiments.commons.save_pop(p, ptdir)
    envs.close()