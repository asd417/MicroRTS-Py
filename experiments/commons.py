import tqdm
import torch
import numpy as np
from typing import Union, Tuple
from torch.utils.tensorboard import SummaryWriter

def write_log(log_message, logdir):
    # Open the log file in append mode ('a'), so new logs are added to the end
    with open(logdir, 'a') as file:
        file.write(log_message + '\n')

def save_pop(p, ptdir):
    torch.save(p, ptdir)

def load_or_create_pop(size, ssvd, logdir, ptdir, device='cpu'):
    #p, logfile = load_checkpoint(name+".pt", name+".txt")
    gi = 1
    try:
        open(ptdir)
        with open(logdir, 'r') as file:
            print(f"Found existing log {logdir}")
            lines = file.readlines()
            last_line = lines[-1].strip().split()
            if last_line[0] == "Generation":
                gi = int(last_line[1])
                print(f"Continuing from Generation {gi}")
        p = torch.load(ptdir)
    except FileNotFoundError:    
        p = create_population(ssvd.get_chromosome_size(), size, device)
        with open(logdir, 'w') as file:
            file.write("Starting new training loop" + '\n')
    return gi, p

def keep_bounds(population: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    minimum = bounds[:, 0] 
    maximum = bounds[:, 1] 
    return torch.clamp(population, min=minimum, max=maximum)


def from_bounds(shape, low, high, device):
    x = torch.randn(shape, device=device)
    return x * (high - low) + low

def create_population(chromosome_len, size, device, bounds=[0,1]):
    return torch.stack([from_bounds(chromosome_len,bounds[0],bounds[1],device) for _ in range(size)])

def apply_fitness(gennum, envs, ssvd, population, func, maxstep, render, maxfit=1000,writer=None, progress=None):
    best_fitness_single_gen = 0
    best_chromosome = None
    ev_f = []
    win = False
    for chromosome in population:
        f = func(envs, chromosome, ssvd, maxstep=maxstep, render=render) 
        tqdm.tqdm.write(f"Fitness: {f}")
        if isinstance(writer,SummaryWriter):
            writer.add_histogram("Chromosome", chromosome, gennum)
        
        if f > best_fitness_single_gen:
            best_fitness_single_gen = f
            best_chromosome = chromosome
        if(f >= maxfit): # we want 90% accuracy of our model's policy
            win = True
        ev_f.append(f)
        if isinstance(writer,tqdm.tqdm):
            progress.update(1)
    return win, np.array(ev_f), best_chromosome, best_fitness_single_gen

def __parents_choice(population: torch.Tensor, n_parents: int) -> torch.Tensor:
    pob_size = population.shape[0]
    choices = torch.arange(pob_size).repeat(pob_size, 1)
    mask = torch.ones_like(choices, dtype=torch.bool)
    mask.fill_diagonal_(False)
    choices = choices[mask].reshape(pob_size, pob_size - 1)
    parents = torch.stack([choices[i, torch.randperm(pob_size - 1)[:n_parents]] for i in range(pob_size)])

    return parents

# See https://github.com/xKuZz/pyade/blob/master/pyade/commons.py#L77

def binary_mutation(population: torch.Tensor, 
                    f: Union[int, float], 
                    bounds: torch.Tensor) -> torch.Tensor:

    # If there's not enough population, return the original population
    if len(population) <= 3:
        return population

    # 1. Select 3 random parents for each individual
    parents = __parents_choice(population, 3)  # Ensure this function is converted to PyTorch

    # 2. Apply the mutation formula: mutated = z + F * (x - y)
    mutated = f * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated += population[parents[:, 2]]

    # 3. Ensure the new population stays within bounds
    return keep_bounds(mutated, bounds)

def current_to_best_2_binary_mutation(population: torch.Tensor,
                                      population_fitness: torch.Tensor,
                                      f: Union[int, float],
                                      bounds: torch.Tensor) -> torch.Tensor:

    # If there's not enough population, return the original population
    if len(population) < 3:
        return population

    # 1. Find the index of the best individual (minimum fitness)
    best_index = torch.argmin(population_fitness)

    # 2. Select two random parents for each individual
    parents = __parents_choice(population, 2)  # Ensure this function is implemented in PyTorch

    # 3. Apply mutation formula
    mutated = population + f * (population[best_index] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    # 4. Ensure new population stays within bounds
    return keep_bounds(mutated, bounds)

def current_to_pbest_mutation(population: torch.Tensor,
                              population_fitness: torch.Tensor,
                              f: list[float],
                              p: Union[float, torch.Tensor, int],
                              bounds: torch.Tensor) -> torch.Tensor:
    """
    "Current to p-best" mutation in differential evolution.
    Mutation formula:
        V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G}) + F * (X_{r1, G} - X_{r2, G})
    """
    if len(population) < 4:
        return population

    # 1. Find the p-best individuals
    p_best = []
    for p_i in p:
        best_index = torch.argsort(population_fitness)[:max(2, round(p_i * len(population)))]
        p_best.append(best_index[torch.randint(len(best_index), (1,)).item()])
    
    p_best = torch.tensor(p_best, dtype=torch.long, device=population.device)

    # 2. Select two random parents
    parents = __parents_choice(population, 2)  # Ensure PyTorch version exists

    # 3. Apply mutation formula
    mutated = population + f * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)

def current_to_rand_1_mutation(population: torch.Tensor,
                               population_fitness: torch.Tensor,
                               k: list[float],
                               f: list[float],
                               bounds: torch.Tensor) -> torch.Tensor:
    """
    "Current to rand/1" mutation in differential evolution.
    Mutation formula:
        U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G}) + F * (X_{r2, G} - X_{r3, G})
    """
    if len(population) <= 3:
        return population

    # 1. Select 3 random parents
    parents = __parents_choice(population, 3)

    # 2. Apply mutation formula
    mutated = k * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])

    return keep_bounds(mutated, bounds)

def current_to_pbest_weighted_mutation(population: torch.Tensor,
                                       population_fitness: torch.Tensor,
                                       f: torch.Tensor,
                                       f_w: torch.Tensor,
                                       p: float,
                                       bounds: torch.Tensor) -> torch.Tensor:
    """
    "Current to p-best weighted" mutation in differential evolution.
    Mutation formula:
        V_{i, G} = X_{i, G} + F_w * (X_{p_best, G} - X_{i, G}) + F * (X_{r1, G} - X_{r2, G})
    """
    if len(population) < 4:
        return population

    # 1. Select p-best individuals
    best_index = torch.argsort(population_fitness)[:max(2, round(p * len(population)))]
    p_best = best_index[torch.randint(len(best_index), (len(population),))]

    # 2. Select two random parents
    parents = __parents_choice(population, 2)

    # 3. Apply mutation formula
    mutated = population + f_w * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)

def crossover(population: torch.Tensor, mutated: torch.Tensor,
              cr: Union[int, float]) -> torch.Tensor:
    """
    Binary crossover operation in differential evolution.
    """
    chosen = torch.rand_like(population)
    j_rand = torch.randint(0, population.shape[1], (population.shape[0],), device=population.device)
    
    row_indices = torch.arange(population.shape[0], device=population.device)
    chosen[row_indices, j_rand] = 0  # Ensures at least one gene is taken from mutated
    
    return torch.where(chosen <= cr, mutated, population)

def exponential_crossover(population: torch.Tensor, mutated: torch.Tensor,
                          cr: Union[int, float]) -> torch.Tensor:
    """
    Exponential crossover operation in differential evolution.
    """
    if isinstance(cr, (int, float)):
        cr = torch.full((population.shape[0],), cr, device=population.device)
    else:
        cr = cr.flatten()

    def __exponential_crossover_1(x: torch.Tensor, y: torch.Tensor, cr: float) -> torch.Tensor:
        z = x.clone()
        n = x.shape[0]
        k = torch.randint(0, n, (1,)).item()
        j = k
        l = 0
        while True:
            z[j] = y[j]
            j = (j + 1) % n
            l += 1
            if torch.randn(1).item() >= cr or l == n:
                return z

    return torch.stack([__exponential_crossover_1(population[i], mutated[i], cr[i]) for i in range(len(population))])

def selection(population: torch.Tensor, new_population: torch.Tensor,
              fitness: torch.Tensor, new_fitness: torch.Tensor,
              return_indexes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given old and new population, select the best individuals based on their fitness.
    """
    indexes = (fitness > new_fitness).nonzero(as_tuple=True)[0]
    population[indexes] = new_population[indexes]
    
    if return_indexes:
        return population, indexes
    else:
        return population

def roulette_selection(population: torch.Tensor, device="cpu"):
    """
    Perform roulette wheel selection on a population.
    
    Args:
        population: List of tuples (chromosome, fitness)
    
    Returns:
        Selected chromosome based on fitness proportionate selection.
    """
    # Extract fitness values
    fitness_values = torch.tensor([fitness for _, fitness in population], dtype=torch.float32, device=device)
    # Handle negative fitness by shifting if needed
    min_fitness = torch.min(fitness_values)
    if min_fitness < 0:
        fitness_values -= min_fitness  # Shift to make all values non-negative

    # Compute selection probabilities
    total_fitness = torch.sum(fitness_values)
    if total_fitness == 0:  # Avoid division by zero
        probabilities = torch.ones_like(fitness_values) / len(fitness_values)
    else:
        probabilities = fitness_values / total_fitness  # Normalize fitness

    # Perform roulette wheel selection
    selected_indices = torch.multinomial(probabilities, 2, replacement=False).tolist()
    return population[selected_indices[0]][0], population[selected_indices[1]][0]  # Return selected chromosome