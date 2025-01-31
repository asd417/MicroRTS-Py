import torch

# Create matrices
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Move to GPU
A = A.to('cuda')
B = B.to('cuda')

# Perform matrix multiplication
result = torch.matmul(A, B)

# Move result back to CPU (if needed)
result_cpu = result.cpu()

print(result_cpu)
# The returned tensors U and V are not unique, 
# nor are they continuous with respect to A. 
# Due to this lack of uniqueness, 
# different hardware and software may compute 
# different singular vectors.
# U, S, Vh = torch.linalg.svd(A)

