# Dependencies required are
import os
import math
import tqdm
import torch # with cuda
import torch.jit

import statistics

import numpy as np
import gymnasium as gym
# if you want to record videos, install stable-baselines3 and use its `VecVideoRecorder`
from stable_baselines3.common.vec_env import VecVideoRecorder

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env_custom import MicroRTSGridModeVecEnv
from gym_microrts.envs.vec_mcts_env import MicroRTSMCTSEnv

from torch.utils.tensorboard import SummaryWriter
import datetime

def create_population(shape, size, device):
    return torch.stack([torch.randn(shape, device=device) for _ in range(size)])

def roulette_wheel_selection(population, device="cpu"):
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

class SSVDVariable:
    def __init__(self, input_w, input_h, outputSize, structure, k='full'):
        self.inputSizeW = input_w
        self.inputSizeH = input_h
        self.outputSize = outputSize
        self.pre_s_tensors = structure[0]
        self.post_s_tensors = structure[1]

    def get_chromosome_size(self):
        return self.pre_s_tensors * min(self.inputSizeH,self.inputSizeW)**2 + self.post_s_tensors * max(self.inputSizeH,self.inputSizeW)**2 + self.outputSize * self.inputSizeW * self.inputSizeH

    def chromosome_to_weights(self, chromosome : torch.Tensor):
        expected_size = self.get_chromosome_size()
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")
        weights_1 = chromosome[
            : self.pre_s_tensors * min(self.inputSizeH,self.inputSizeW)**2
            ].view(self.pre_s_tensors, min(self.inputSizeH,self.inputSizeW), min(self.inputSizeH,self.inputSizeW))      # First matrix (n x n)
        weights_2 = chromosome[
            self.pre_s_tensors * min(self.inputSizeH,self.inputSizeW)**2 : 
            self.pre_s_tensors * min(self.inputSizeH,self.inputSizeW)**2 + self.post_s_tensors * max(self.inputSizeH,self.inputSizeW)**2
            ].view(self.post_s_tensors, max(self.inputSizeH,self.inputSizeW), max(self.inputSizeH,self.inputSizeW))   # Second matrix (m x n^2)
        weightO = chromosome[self.pre_s_tensors * min(self.inputSizeH,self.inputSizeW)**2 + self.post_s_tensors * max(self.inputSizeH,self.inputSizeW)**2 : 
                             ].view(self.outputSize, self.inputSizeW * self.inputSizeH)
        return weights_1, weights_2, weightO

class SSVD:
    def __init__(self, input_w, input_h, outputSize, k='full'):
        self.inputSizeW = input_w
        self.inputSizeH = input_h
        self.outputSize = outputSize

    def get_chromosome_size(self):
        return min(self.inputSizeH,self.inputSizeW)**2 + self.outputSize * self.inputSizeW * self.inputSizeH
        
    def chromosome_to_weights(self, chromosome : torch.Tensor):
        expected_size = self.get_chromosome_size()
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")
        weights1 = chromosome[: min(self.inputSizeH,self.inputSizeW)**2].view(min(self.inputSizeH,self.inputSizeW), min(self.inputSizeH,self.inputSizeW))      # First matrix (n x n)
        weightsO = chromosome[min(self.inputSizeH,self.inputSizeW)**2:].view(self.outputSize, self.inputSizeW * self.inputSizeH)   # Second matrix (m x n^2)
        return weights1, weightsO
    
def evaluateSSVD_simple(weights1,weightsO,input : torch.Tensor) -> torch.Tensor:
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

def evaluateSSVD(weights1,weights2,weightsO,input : torch.Tensor) -> torch.Tensor:
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

    result = torch.nn.functional.relu(U_stable @ weights1[0])
    for i in range(1, weights1.shape[0]):
        result = torch.nn.functional.relu(result @ weights1[i])  # ReLU after each step
    result = torch.nn.functional.relu(result @ Sigma)
    for i in range(1, weights2.shape[0]):
        result = torch.nn.functional.relu(result @ weights2[i])  # ReLU after each step
    result = weightsO @ (result @ Vh_stable).flatten()

    #outputTensor = weightsO @ (U_stable @ weights1 @ Sigma @ Vh_stable).flatten()
    return result

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

#TODO let this spawn multiple environments at the same time. The microrts client already supports it.
#need to revert some of the work I did on the java side because I might have broken it.
def start_game(envs, weights1, weights2, weightsO, seed, device="cpu", maxstep=10000):
    obs = envs.reset()
    reward_sum = 0
    for i in range(maxstep):
        if RENDER:
            if RECORD:
                envs.render(mode="rgb_array")
            else:
                envs.render()
        # to process multiple processes at the same time, dont squeeze. just calculate individually
        inputTensor = torch.from_numpy(obs).to(device).float().squeeze()

        feature_sizes = [5, 5, 3, 8, 6, 2]
        # Extract and process each feature
        p = 0
        features = []
        for size in feature_sizes:
            feature = inputTensor[:, :, p:p + size]
            weights = torch.arange(size, device=inputTensor.device).reshape(1, 1, size)
            features.append((feature * weights).sum(dim=2, keepdim=True))
            p += size

        # Concatenate processed features
        inputTensor = torch.cat(features, dim=2).unsqueeze(0).unsqueeze(0)
        #print(inputTensor.shape)
        m = torch.nn.Conv3d(1, 1, (1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 2))
        m.to(inputTensor.device)
        it : torch.Tensor = m(inputTensor)
        it = m(it)
        it = m(it)
        m = torch.nn.Conv3d(1, 1, (1, 1, 2), stride=(1, 1, 1), padding=(0, 0, 0))
        m.to(inputTensor.device)
        it = m(it)
        it = it.squeeze()
        outputTensor = evaluateSSVD(weights1,weights2,weightsO,it)
        del inputTensor
        # sample valid actions
        num_unit_type = 6
        # Concatenate slices of outputTensor
        action = outputTensor.to("cpu").detach().numpy()
        # doing the following could result in invalid actions

        # to run multiple environments, wrap this in array and combine with other actions
        # action = np.array([envs.action_space.sample()])
        obs, reward, done, info = envs.step(action)
        #print(done)
        #print(reward)
        reward_sum += sum(reward)
        if done.any():
            return reward_sum
    return reward_sum

def fitness(envs, chromosome, ssvd, device, trials=10):
    # chromosome is a 1D vector
    weights1, weights2, weightsO = ssvd.chromosome_to_weights(chromosome)
    fits = [start_game(envs, weights1, weights2, weightsO, x, device=device) for x in range(trials)]
    del weights1
    del weights2
    del weightsO
    return (sum(fits) + 10 * trials)/float(trials)

def start_game_mcts(envs, chromosome, device="cpu", maxstep=10000):
    envs.reset(chromosome)
    reward_sum = 0
    for i in range(maxstep):
        if RENDER:
            if RECORD:
                envs.render(mode="rgb_array")
            else:
                envs.render()
        _, reward, done, info = envs.step()
        reward_sum += sum(reward)
        if done.any():
            return reward_sum
    return reward_sum

def fitness_mcts(envs, chromosome, ssvd, device, trials=10):
    # chromosome is a 1D vector
    fits = [start_game_mcts(envs, chromosome, device=device) for x in range(trials)]
    return sum(fits) + 10

def load_files(pt_file, txt_file):
    # Check if the .pt file exists
    if os.path.exists(pt_file):
        tensor_data = torch.load(pt_file)
        print(f"Loaded tensor data from {pt_file}")
    else:
        tensor_data = None
        print(f"{pt_file} does not exist.")
    
    # Check if the .txt file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            text_data = file.read()
        print(f"Loaded text data from {txt_file}")
    else:
        text_data = None
        print(f"{txt_file} does not exist.")
    
    return tensor_data, text_data

def write_log(log_message, name="population"):
    # Open the log file in append mode ('a'), so new logs are added to the end
    with open(name+"_log.txt", 'a') as file:
        file.write(log_message + '\n')
    #print(f"Log written to {log_file}")

# input is (1, 16, 16, 29)
def dout(din,kernel_size, padding=0,dilation=1, stride=1):
    return math.floor((din + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def test_conv():
    # m = torch.nn.Conv3d(1, 1, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = torch.nn.Conv3d(1, 1, (1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 2))
    input = torch.randn(1, 1, 16, 16, 29)
    output : torch.Tensor = m(input)
    print(input.shape)
    print(output.shape)
    output = m(output)
    print(output.shape)
    output = m(output)
    print(output.shape)
    output = m(output)
    m = torch.nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0))
    print(output.shape)
    output = m(output)
    print(output.shape)

    output = output.squeeze(-1)
    print(output.shape)

    #print(dout(29, 5, stride=3, padding=2))
    return

def save_pop(p, name="population"):
    torch.save(p, name+'.pt')

def load_or_create_pop(size, name="population"):
    p, logfile = load_files(name+".pt", name+"_log.txt")
    gi = 1
    if p is None:
        p = create_population((ssvd.get_chromosome_size(),1), size, device)
        with open(name+"_log.txt", 'w') as file:
            file.write("Starting new training loop" + '\n')

    if not logfile is None:
        with open(name+"_log.txt", 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip().split()
            if last_line[0] == "Generation":
                gi = int(last_line[1])
                print(f"Continuing from Generation {gi}")
    return gi, p

def get_logger(name, directory="runs/") -> SummaryWriter:
    log_dir = f"{directory}{name}"
    writer = SummaryWriter(log_dir)
    return writer

# openai es
def run_test_es(ssvd, envs, trials, pop_size, max_iter, device, fitness_func, name="OpenAI-ES"):
    test_name = name + "-population"
    sigma = 0.1    # noise standard deviation
    alpha = 0.001  # learning rate
    gen_start, w = load_or_create_pop(1, name=test_name)
    writer = get_logger(name)
    
    for i in range(gen_start, max_iter):
        w = w.squeeze(0)
        progress = tqdm.tqdm(total=pop_size, desc=f"{test_name} Generation {i}/{max_iter}")
        N = torch.randn((pop_size, ssvd.get_chromosome_size(), 1), device=device)
        R = torch.zeros(pop_size)
        best_fitness_single_gen = 0

        for j in range(pop_size):
            w_try = w + sigma*N[j]
            f = fitness_func(envs, w_try, ssvd, device, trials=trials)
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
        write_log(logstr, name=test_name)
        A = (R - torch.mean(R)) / torch.std(R)
        N = N.squeeze()
        w = w.squeeze(-1)
        #print(f"N.T @ A {(N.T @ A).shape}")
        #print(f"w {w.shape}")
        w = w + alpha/(pop_size*sigma) * N.T @ A
        w = w.unsqueeze(0).unsqueeze(-1)
        save_pop(w, name=test_name)


def run_test_ga(ssvd, envs, trials, pop_size, max_iter, device, fitness_func, name="GA", elitism=0.1):
    test_name = name + "-population"
    writer = get_logger(name)
    gi, p = load_or_create_pop(pop_size, name=test_name)
    mutation_rate = 0.5

    best_chromosome = None
    best_fitness = 0
    win = False
    while win == False:
        if(max_iter < gi):
            win = True
            break
        ev_f = []
        best_fitness_single_gen = 0
        progress = tqdm.tqdm(total=len(p), desc=f"{test_name} Generation {gi}/{max_iter}")
        for chromosome in p:
            f = fitness_func(envs, chromosome, ssvd, device, trials=trials) 
            tqdm.tqdm.write(f"Fitness: {f}")
            writer.add_histogram("Chromosome", chromosome, gi)
            
            if f > best_fitness_single_gen:
                best_fitness_single_gen = f
            if f > best_fitness:
                best_fitness = f
                best_chromosome = chromosome
            if(f >= 1000 * 0.9): # we want 90% accuracy of our model's policy
                win = True
            ev_f.append(f)
            progress.update(1)
        avg = sum(ev_f) / len(ev_f)
        std = statistics.stdev(ev_f)
        writer.add_scalars(f"{name}/Fitness", {
            "Best Fitness": best_fitness_single_gen,
            "Average Fitness": avg,
            "Standard Deviation": std,
            "Upper Bound": avg + std,
            "Lower Bound": avg - std
        }, gi)
        logstr = f"Generation {gi} {name} Highest: {best_fitness_single_gen} Average: {avg} StDev: {std}"
        tqdm.tqdm.write(logstr)
        write_log(logstr, name=test_name)
        if(not win):
            ev_p = list(zip(p,ev_f))
            ev_p_sorted = sorted(ev_p, key=lambda x: x[1], reverse=True) # sort by fitness from highest to lowest
            elites = int(pop_size * elitism)
            ev_p_sorted = ev_p_sorted[:elites]
            new_p = []
            print("Preparing next generation...")
            for i in range(pop_size - elites): # GA
                parent1, parent2 = roulette_wheel_selection(ev_p, device)
                #print("Crossover...")
                co = crossover(parent1, parent2)
                #print("Mutating...")
                mutate = mutate_multivariate_gaussian(co, mutation_rate)
                #print("Done Individual Creation")
                new_p.append(mutate)
                
            p = [tup[0] for tup in ev_p_sorted] + new_p
            gi += 1
        else:
            logstr = f"Training Done | Best Fitness: {best_fitness}"
            tqdm.tqdm.write(logstr)
            write_log(logstr, name=test_name)
            logstr = f"Chromosome: {chromosome}"
            tqdm.tqdm.write(logstr)
            write_log(logstr, name=test_name)
            save_pop(best_chromosome, name=test_name+"_best")
        save_pop(p, name=test_name)
    envs.close()

RECORD = False
RENDER = False
USE_MCTS = False
if __name__ == "__main__":
    #test_conv()
    if not USE_MCTS:
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        fitness_f = fitness
    else:
        envs = MicroRTSMCTSEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        fitness_f = fitness_mcts
    if RECORD:
        envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    input_h = envs.height
    input_w = envs.width
    actionSpace = envs.height * envs.width + 6 # board + unit type count
    print(f"Observation Space Height: {input_h}")
    print(f"Observation Space Width: {input_w}")
    print(f"Action Space size: {actionSpace}")

    ssvd = SSVDVariable(input_w, input_h, actionSpace, [2,2])
    #run_test_ga(ssvd, envs, 10, 100, 300, device, name="GA_100_10%", elitism=0.1)
    run_test_ga(ssvd, envs, 5, 100, 300, device, fitness_f, name="GA_5_100_10%", elitism=0.1)
    #run_test_es(ssvd, envs, 10, 50, 300, device)

if __name__ == "__main__1":
    # writer = get_logger("test")
    # writer.add_scalar("Loss/train", 0.1, 1)  # ✅ Logs loss over time
    # writer.add_scalar("Loss/train", 0.4, 2)
    # writer.add_scalar("Loss/train", 0.5, 3)
    # sample_input = torch.randn(1, 3, 32, 32)  
    # writer.add_histogram("histor", sample_input, 1)
    # sample_input = torch.randn(1, 3, 32, 32)
    # writer.add_histogram("histor", sample_input, 2)
    # sample_input = torch.randn(1, 3, 32, 32)
    # writer.add_histogram("histor", sample_input, 3)
    # writer.close()
    input_w = 10
    input_h = 12
    output = 8
    layers = [3,2]
    ssvd = SSVDVariable(input_w, input_h, output, layers)
    t = torch.randn(ssvd.get_chromosome_size())
    w1, w2, w3 = ssvd.chromosome_to_weights(t)
    t1 = torch.randn((input_w,input_h))
    U, S, Vh = torch.linalg.svd(t1)
    Sigma = torch.zeros(t1.shape)
    Sigma[:, :S.size(0)] = torch.diag(S)
    print(ssvd.get_chromosome_size())
    print(U.shape)
    print(w1.shape)
    print(Sigma.shape)
    print(w2.shape)
    print(Vh.shape)
    print(w3.shape)
    result = evaluateSSVD(w1, w2, w3, t1)
    print(result.shape)