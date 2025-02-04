# Example code of SSVD trying to learn an arbitrary function
# Dependencies required are
import tqdm
import torch # with cuda
import torch.jit

import statistics

import numpy as np
import gymnasium as gym
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSGridModeSharedMemVecEnv

def create_population(shape, size, device):
    p = []
    for i in range(size):
        p.append(torch.randn(shape, device=device))
    return p

def roulette_wheel_selection(population, device='cpu'):
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

def crossover(parent1, parent2):
    """Performs single-point crossover between two matrices."""
    rows, _ = parent1.shape
    crossover_pointr = torch.randint(0, rows, (1,),device=parent1.device).item()  # Choose a row to swap from
    child = torch.clone(parent1)
    # Create a child matrix by swapping rows below crossover point
    child[crossover_pointr:, :] = parent2[crossover_pointr:, :]  # Swap lower part
    return child

def mutate_multivariate_gaussian(matrix, mutation_rate=0.1):
    """
    Applies mutation by adding noise sampled from a multivariate Gaussian distribution.
    The mean is zero, and the covariance matrix is an identity matrix scaled by a factor.
    """
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

# Used for generating classification dataset. make sure to set the torch.manual_seed
def matrix_to_vector_custom(matrix: torch.Tensor, m : int, seed : int):
    """
    Maps an (n x n) matrix to a unique (m x 1) vector using a deterministic transformation.

    Args:
        matrix (torch.Tensor): Input tensor of shape (n, n).
        m (int): Desired output dimension (must be <= n*n for uniqueness).

    Returns:
        torch.Tensor: Transformed vector of shape (m, 1).
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square (n x n).")
    if m > n * n:
        raise ValueError(f"m must be <= {n*n} to maintain uniqueness.")

    # Flatten the matrix to shape (n*n, 1)
    vector = matrix.view(n * n, 1)
    # make deterministic
    torch.manual_seed(seed)
    # Create a fixed transformation matrix W of shape (m, n*n)
    W = torch.randn(m, n * n, device=matrix.device)  # Ensure full rank
    torch.manual_seed(torch.randint(0, 1000, (1,)).item())
    # Compute transformed vector
    transformed_vector = W @ vector
    return transformed_vector

class SSVD:
    def __init__(self, input_w, input_h, outputSize, k='full'):
        self.inputSizeW = input_w
        self.inputSizeH = input_h
        self.outputSize = outputSize

    def get_chromosome_size(self):
        return min(self.inputSizeH,self.inputSizeW)**2 + self.outputSize * self.inputSizeW * self.inputSizeH
        
    def chromosome_to_weights(self, chromosome):
        expected_size = self.get_chromosome_size()
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")

        weights1 = chromosome[: min(self.inputSizeH,self.inputSizeW)**2].view(min(self.inputSizeH,self.inputSizeW), min(self.inputSizeH,self.inputSizeW))      # First matrix (n x n)
        weightsO = chromosome[min(self.inputSizeH,self.inputSizeW)**2:].view(self.outputSize, self.inputSizeW * self.inputSizeH)   # Second matrix (m x n^2)
        return weights1, weightsO
    
def evaluateSSVD(weights1,weightsO,input : torch.Tensor):
    input = input.float() # ensure that the input is floating point tensor
    U, S, Vh = torch.linalg.svd(input)
    Sigma = torch.zeros(input.shape, device=input.device)
    Sigma[:, :S.size(0)] = torch.diag(S)
    # Apply QR decomposition to stabilize U and Vh
    U_stable, _ = torch.linalg.qr(U)  # QR decomposition of U
    Vh_stable, _ = torch.linalg.qr(Vh.T)  # QR decomposition of Vh.T, then transpose back
    #print(f"{input.shape} at {input.device}")
    #print(f"{U_stable.shape} at {U_stable.device}")
    #print(f"{weights1.shape} at {weights1.device}")
    #print(f"{Sigma.shape} at {Sigma.device}")
    #print(f"{Vh_stable.shape} at {Vh_stable.device}")
    #print(f"{weightsO.shape} at {weightsO.device}")

    #outputTensor = weightsO @ (Vh_stable @ Sigma @ weights1 @ U_stable).flatten()
    outputTensor = weightsO @ (U_stable @ weights1 @ Sigma @ Vh_stable).flatten()
    return outputTensor

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)

def start_game(envs, weights1, weightsO, seed, device="cpu", maxstep=10000):
    envs.action_space.seed(seed)
    obs = envs.reset()
    reward_sum = 0
    for i in range(maxstep):
        if RENDER:
            if RECORD:
                envs.render(mode="rgb_array")
            else:
                envs.render()
        action_mask = envs.get_action_mask()
        #print(f"Action Mask Shape: {action_mask.shape}")
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        print(obs)
        inputTensor = torch.from_numpy(obs.reshape(envs.height, -1)).to(device)
        #print(inputTensor)
        outputTensor = evaluateSSVD(weights1,weightsO,inputTensor)
        del inputTensor
        outputTensor = outputTensor.reshape(-1, action_mask.shape[-1])
        outputTensor[action_mask == 0] = -9e8 # mask action with action mask
        # sample valid actions
        action = np.concatenate(
            (
                sample(action_mask[:, 0:6]),  # action type
                sample(action_mask[:, 6:10]),  # move parameter
                sample(action_mask[:, 10:14]),  # harvest parameter
                sample(action_mask[:, 14:18]),  # return parameter
                sample(action_mask[:, 18:22]),  # produce_direction parameter
                sample(action_mask[:, 22:29]),  # produce_unit_type parameter
                # attack_target parameter
                sample(action_mask[:, 29 : sum(envs.action_space.nvec[1:])]),
            ),
            axis=1,
        )
        
        # doing the following could result in invalid actions
        # action = np.array([envs.action_space.sample()])
        obs, reward, done, info = envs.step(action)
        #print(done)
        #print(reward)
        reward_sum += sum(reward)
        if done.any():
            return reward_sum
    return reward_sum

def fitness(envs, chromosome, ssvd, device, trials=1):
    # chromosome is a 1D vector
    weights1, weightsO = ssvd.chromosome_to_weights(chromosome)
    fits = [start_game(envs, weights1, weightsO, x, device=device) for x in range(trials)]
    del weights1
    del weightsO
    return sum(fits)

if __name__ == "__main__":
    RECORD = False
    RENDER = True
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(1)],
        map_paths=["maps/16x16/basesWorkers16x16.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    if RECORD:
        envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    population = 20
    input_h = envs.height
    input_w = envs.width * sum(envs.num_planes)
    actionSpace = sum(envs.action_space.nvec)
    print(f"Observation Space Height: {input_h}")
    print(f"Observation Space Width: {input_w}")
    print(f"Action Space size: {actionSpace}")

    ssvd = SSVD(input_w, input_h, actionSpace)
    p = create_population((ssvd.get_chromosome_size(),1), population, device)
    mutation_rate = 0.5
    win = False
    gi = 1
    maxgen = 1000
    
    best_chromosome = None
    best_fitness = 0
    while win == False:
        if(maxgen < gi):
            win = True
            break
        #print(f"Generation{gi}")
        ev_f = []
        progress = tqdm.tqdm(total=len(p), desc=f"Generation {gi}/{maxgen}")
        
        for chromosome in p:
            f = fitness(envs, chromosome, ssvd, device)
            tqdm.tqdm.write(f"Fitness: {f}")
            if f > best_fitness:
                best_fitness = f
                best_chromosome = chromosome
            if(f >= 1000 * 0.9): # we want 90% accuracy of our model's policy
                win = True
            ev_f.append(f)
            progress.update(1)
        avg = sum(ev_f) / len(ev_f)
        tqdm.tqdm.write(f"Generation {gi} Average: {avg} StDev: {statistics.stdev(ev_f)}")
        if(not win):
            ev_p = list(zip(p,ev_f))
            ev_p_sorted = sorted(ev_p, key=lambda x: x[1], reverse=True) # sort by fitness from highest to lowest
            elitism = int(population * 0.9) # 90 percent of the networks are preserved
            ev_p_sorted = ev_p_sorted[:elitism]
            new_p = []
            for i in range(population - elitism): # GA
                parent1, parent2 = roulette_wheel_selection(ev_p)
                new_p.append(mutate_multivariate_gaussian(crossover(parent1, parent2), mutation_rate))
            p = [tup[0] for tup in ev_p_sorted] + new_p
            gi += 1
        else:
            tqdm.tqdm.write(f"Training Done | Best Fitness: {best_fitness}")
            tqdm.tqdm.write(f"Chromosome: {chromosome}")
    envs.close()