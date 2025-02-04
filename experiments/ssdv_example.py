# Example code of SSVD trying to learn an arbitrary function
# Dependencies required are
import tqdm
import torch # with cuda
import torch.jit

import statistics

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
    def __init__(self, inputSideLength, outputSize):
        self.inputSize = inputSideLength
        self.outputSize = outputSize
        
    def chromosome_to_weights(self, chromosome):
        expected_size = self.inputSize * self.inputSize + self.outputSize * self.inputSize * self.inputSize
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")

        weights1 = chromosome[: self.inputSize * self.inputSize].view(self.inputSize, self.inputSize)      # First matrix (n x n)
        weightsO = chromosome[self.inputSize * self.inputSize:].view(self.outputSize, self.inputSize * self.inputSize)   # Second matrix (m x n^2)
        return weights1, weightsO
    
def evaluateSSVD(weights1,weightsO,input):
    U, S, Vh = torch.linalg.svd(input)
    # Apply QR decomposition to stabilize U and Vh
    U_stable, _ = torch.linalg.qr(U)  # QR decomposition of U
    Vh_stable, _ = torch.linalg.qr(Vh.T)  # QR decomposition of Vh.T, then transpose back
    outputTensor = weightsO @ (Vh_stable @ torch.diag(S) @ weights1 @ U_stable).flatten()
    probabilities = torch.softmax(outputTensor, dim=0)
    sampled_action = torch.multinomial(probabilities, num_samples=1).item()
    return sampled_action

# example of a single policy
def single_policy(weights1, weightsO, inputSize : int, outputSize : int, seed : int):
    # calculate theoratically optimal action
    # create a random state
    inputVector = torch.randn_like(torch.ones(inputSize, inputSize), dtype=torch.float32, device=weights1.device)
    # by a deterministic rule, the random state is transformed into an output
    # the network needs to learn the rule
    targetOutputTensor = matrix_to_vector_custom(inputVector, outputSize, seed)
    # get optimal policy
    target_index = torch.argmax(targetOutputTensor).item() 

    sampled_action = evaluateSSVD(weights1, weightsO, inputVector)
    if target_index == sampled_action:
        return 1
    return 0

# use (outputsize,flattened) matrix times (flattened, 1) to calculate output tensor
# need to adjust the split in the fitness function as well

# emulate gameplay by not using loss function (we dont want to be doing behavior cloning) 
# and calculating fitness at the end of the evaluation. 
# (softmax both the output tensor and target tensor, sample and if same, score 1: do this 1000 times)

# assumes two weights matrices nxn and mx1
def fitness(chromosome, ssvd, trials=500):
    # chromosome is a 1D vector
    weights1, weightsO = ssvd.chromosome_to_weights(chromosome)
    futures = [torch.jit.fork(single_policy, weights1, weightsO, ssvd.inputSize, ssvd.outputSize, x) for x in range(trials)]
    fits = [torch.jit.wait(fut) for fut in futures]
    return sum(fits)

import torch.nn as nn
class FitnessModel(nn.Module):
    def __init__(self,i,o):
        self.inputSize : int = i
        self.outputSize : int = o
        self.trials : int = 500
        super().__init__()

    def forward(self, chromosome):
        weights1, weightsO = self.chromosome_to_weights(chromosome)
        futures = [torch.jit.fork(single_policy, weights1, weightsO, self.inputSize, self.outputSize, x) for x in range(self.trials)]
        fits = [torch.jit.wait(fut) for fut in futures]
        return sum(fits)

    def chromosome_to_weights(self, chromosome):
        expected_size = self.inputSize * self.inputSize + self.outputSize * self.inputSize * self.inputSize
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")

        weights1 = chromosome[: self.inputSize * self.inputSize].view(self.inputSize, self.inputSize)      # First matrix (n x n)
        weightsO = chromosome[self.inputSize * self.inputSize:].view(self.outputSize, self.inputSize * self.inputSize)   # Second matrix (m x n^2)
        return weights1, weightsO

if __name__ == "__main__":
    USETORCHSCRIPT = False #doesnt seem to make any difference on my computer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    population = 20
    inputShape_n = 30 # length of one side of the input
    actionSpace = 20 # 
    p = create_population((inputShape_n * inputShape_n + actionSpace * inputShape_n * inputShape_n,1), population, device)
    mutation_rate = 0.5
    win = False
    gi = 1
    maxgen = 1000
    ssvd = SSVD(inputShape_n, actionSpace)
    if USETORCHSCRIPT:
        scripted_model = torch.jit.script(FitnessModel(inputShape_n, actionSpace))
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
            f = 0
            if USETORCHSCRIPT:
                f = scripted_model(chromosome)
            else:
                f = fitness(chromosome, ssvd)
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