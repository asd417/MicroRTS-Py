import torch
from torch import nn
class SSVDVariable:
    # U is of size (input_h,input_h)
    # Vh is of size (input_w,input_w)
    # therefore
    # weights1[0] is of size (input_h, input_h)
    # weights2[0] is of size (input_w, input_w)
    def __init__(self, input_h, input_w, outputSize, structure, k='full'):
        self.inputSizeW = input_w
        self.inputSizeH = input_h
        self.outputSize = outputSize
        self.pre_s_tensors = structure[0]
        self.post_s_tensors = structure[1]
        if k != "full":
            self.k = k
        else:
            self.k = max(input_w, input_h)

    def get_chromosome_size(self):
        return self.pre_s_tensors * min(self.inputSizeH, self.k) * min(self.inputSizeH, self.k) + self.post_s_tensors * min(self.inputSizeW, self.k) * min(self.inputSizeW, self.k) + self.outputSize * self.inputSizeW * self.inputSizeH

    def chromosome_to_weights(self, chromosome : torch.Tensor):
        expected_size = self.get_chromosome_size()
        if chromosome.shape[0] != expected_size:
            raise ValueError(f"Vector size must be {expected_size}, but got {chromosome.shape[0]}.")
        slice1 = self.pre_s_tensors * min(self.inputSizeH, self.k)**2

        slice2 = self.post_s_tensors * min(self.inputSizeW, self.k)**2

        weights_1 = chromosome[ : slice1].view(self.pre_s_tensors, min(self.inputSizeH, self.k), min(self.inputSizeH, self.k))      # First matrix (n x n)
        
        weights_2 = chromosome[slice1 : slice1 + slice2].view(self.post_s_tensors,  min(self.inputSizeW, self.k), min(self.inputSizeW, self.k))   # Second matrix (m x n^2)
        weightO = chromosome[slice1 + slice2 : ].view(self.outputSize, self.inputSizeW * self.inputSizeH)
        return weights_1, weights_2, weightO


class SSVDModel(nn.Module):
    def __init__(self, envs, k, device="cpu"):
        super(SSVDModel, self).__init__()
        self.envs = envs
        self.feature_sizes = [5, 5, 3, 8, 6, 2]
        
        # Define convolution layers
        self.conv1 = nn.Conv3d(1, 1, (1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 2), device=device)
        self.conv2 = nn.Conv3d(1, 1, (1, 1, 2), stride=(1, 1, 1), padding=(0, 0, 0), device=device)
        self.k = k

    def forward(self, obs, weights1, weights2, weightsO):
        inputTensorMulti = torch.from_numpy(obs).float()
        batch_size = self.envs.num_envs
        device = inputTensorMulti.device
        H, W, D = inputTensorMulti.shape[1:]  # Get spatial dimensions

        # Step 1: Extract Features for All Environments at Once
        feature_tensors = []
        p = 0
        
        assert D == sum(self.feature_sizes), "Depth should be equal to the sum of all feature planes"
        for size in self.feature_sizes:
            feature = inputTensorMulti[:, :, :, p:p + size]  # Extract along the last dimension
            #print(f"shape: {feature.shape}")
            weights = torch.arange(size, device=device).reshape(1, 1, 1, size)
            #print(f"shape: {weights}")
            f = (feature * weights).sum(dim=3, keepdim=True)
            #print(f"shapef: {f.shape}")
            feature_tensors.append(f)  # Sum along feature axis
            #print(f)
            p += size
        
        # Step 2: Stack extracted features into a 5D tensor (batch, channels=1, depth=1, height=H, width=W)
        inputTensor = torch.cat(feature_tensors, dim=3) # Shape: (batch, H, W, D_n)
        s1 = inputTensor.shape
        inputTensor = inputTensor.unsqueeze(1) # Shape: (batch, 1, H, W, D_n)
        
        # Step 3: Pass through convolutions
        it = self.conv1(inputTensor)
        it = self.conv1(it)
        it = self.conv1(it)
        it = self.conv2(it)
        s2 = it.shape
        # Step 4: Squeeze the depth dimension (since it's 1)
        it = it.squeeze(-1)  # Shape: (batch, H, W)
        
        s3 = it.shape
        # Step 5: Process each environment separately through evaluateSSVD
        actions = []
        for i in range(batch_size):
            s4 = it[i][1:].shape # next line squeezes it
            outputTensor = self.evaluateSSVD(weights1, weights2, weightsO, it[i].squeeze(0))  # Process each separately
            #outputTensor[outputTensor < 0] = 0.00001
            actions.append(outputTensor.unsqueeze(0))  # Keep batch dimension
        out = torch.cat(actions, dim=0)
        #print(f"from {s1} to {s2} to {s3} to {s4} to {out.shape}")
        # Step 6: Concatenate the actions into a final output tensor
        return out # Shape: (batch, output_size)
        
    def evaluateSSVD(self, weights1, weights2, weightsO, input : torch.Tensor) -> torch.Tensor:
        input = input.float() # ensure that the input is floating point tensor
        U, S, Vh = torch.linalg.svd(input)
        S_height = S.shape[0]
        S = S[:self.k]
        if S.shape[0] < S_height: # use top-k only
            Sigma = torch.diag(S)
            U = U[:, :self.k]
            #print(Vh.shape)
            Vh = Vh[:self.k, :]
            #print(Vh.shape)
            #print("------------")
        else:
            Sigma = torch.zeros(input.shape, device=input.device) # use full
            Sigma[:, :S.size(0)] = torch.diag(S)
        # Apply QR decomposition to stabilize U and Vh
        U_stable, _ = torch.linalg.qr(U)  # QR decomposition of U
        Vh_stable, _ = torch.linalg.qr(Vh.T)  # QR decomposition of Vh.T, then transpose back
        Vh_stable = Vh_stable.T
        
        result = torch.nn.functional.relu(U_stable @ weights1[0])
        for i in range(1, weights1.shape[0]):
            result = torch.nn.functional.relu(result @ weights1[i])  # ReLU after each step
        result = torch.nn.functional.relu(result @ Sigma)
        for i in range(1, weights2.shape[0]):
            result = torch.nn.functional.relu(result @ weights2[i])  # ReLU after each step
        #test = U_stable @ Sigma @ Vh_stable
        #print(input)
        #print(test)
        #print(U_stable.shape)
        #print(Sigma.shape)
        #print(result.shape)
        #print(Vh_stable.shape)
        #print((result @ Vh_stable).flatten().shape)
        #print(weightsO.shape)
        result = weightsO @ (result @ Vh_stable).flatten()

        return result
    