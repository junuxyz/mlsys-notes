import torch

def naive_softmax(z, K):
    out = torch.empty_like(z)
    denominator = 0

    # Pass 1. compute denominator
    for i in range(K):
        out[i] = torch.exp(z[i])
        denominator += out[i]
    
    # Pass 2. divide each out with denominator
    for i in range(K):
        out[i] = out[i] / denominator
    
    return out


def safe_softmax(z, K):
    out = torch.empty_like(z)
    denominator = 0
    max_K = z[0]

    # Pass 1. get global max
    for i in range(K):
        max_K = max(max_K, i)
    
    # Pass 2. compute denominator
    for i in range(K):
        out[i] = torch.exp(z[i]-max_K)
        denominator += out[i]

    # Pass 3. divide each out[i]-max with denominator
    for i in range(K):
        out[i] = out[i] / denominator

    return out


def online_softmax(z, K):
    out = torch.empty_like(z)
    denominator = 0
    max_i = z[0]

    # Pass 1. compute denominator iteratively
    for i in range(K):
        new_max = max(max_i, z[i])
        denominator = (
            denominator * torch.exp(max_i - new_max)
            + torch.exp(z[i] - new_max)
        )
        max_i = new_max
    
    # Pass 2. divide each element-max with denominator
    for i in range(K):
        # recalculate because max may be changed
        out[i] = torch.exp(z[i]-max_i) / denominator

    return out


def blocked_softmax(z, K, block_size):
    out = torch.empty_like(z)

    global_max = z[0]
    global_denominator = 0

    # Pass 1: compute global max + denominator block by block
    for block_start in range(0, K, block_size):
        block_end = min(block_start + block_size, K)
        
        block_max = z[block_start]
        for i in range(block_start, block_end):
            block_max = max(block_max, z[i])
        
        block_denominator = 0
        for i in range(block_start, block_end):
            block_denominator += torch.exp(z[i]-block_max)
        
        new_max = max(global_max, block_max)
        global_denominator = (
            global_denominator * torch.exp(global_max - new_max)
            + block_denominator * torch.exp(block_max - new_max)
        )
        global_max = new_max

    # Pass 2: produce final softmax
    for i in range(K):
        out[i] = torch.exp(z[i] - global_max) / global_denominator
    
    return out

if __name__ == "__main__":
    z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K = z.shape[0]

    print(f"torch.softmax: {torch.softmax(z, dim=0)} sum: {sum(torch.softmax(z, dim=0))}")
    print(f"naive_softmax: {naive_softmax(z, K)} sum: {sum(naive_softmax(z, K))}")
    print(f"safe_softmax: {safe_softmax(z, K)} sum: {sum(safe_softmax(z, K))}")
    print(f"online_softmax: {online_softmax(z, K)} sum: {sum(online_softmax(z, K))}")
    print(f"blocked_softmax: {blocked_softmax(z, K, 3)} sum: {sum(blocked_softmax(z, K, 3))}")

# % python softmax.py
# torch.softmax: tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]) sum: 0.9999998807907104
# naive_softmax: tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]) sum: 1.0
# safe_softmax: tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]) sum: 0.9999999403953552
# online_softmax: tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]) sum: 1.0
# blocked_softmax: tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]) sum: 1.0