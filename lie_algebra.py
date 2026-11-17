"""
Bryce Bernstein
---------------
This file is a wrapper for lie algebra functions within DPC-Net.
I used https://arxiv.org/pdf/2103.15980 for help understanding matrix
exponentials and logarithms for developing this file.
"""
import torch
import math


def skew_matrix(rotations: torch.Tensor):
    matrices = torch.zeros((rotations.size(0), 3, 3))
    matrices[:, 0, 1] = -rotations[:, 2]
    matrices[:, 0, 2] = rotations[:, 1]
    matrices[:, 1, 0] = rotations[:, 2]
    matrices[:, 1, 2] = -rotations[:, 0]
    matrices[:, 2, 0] = -rotations[:, 1]
    matrices[:, 2, 1] = rotations[:, 0]
    return matrices

def lie_hat(vectors: torch.Tensor):
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)   
    num_vecs = vectors.size(0)
    
    trans = vectors[:, :3]
    rots = vectors[:, 3:]
    w_h = skew_matrix(rots)
    w_h2 = w_h @ w_h
    
    theta = torch.linalg.vector_norm(rots, dim=1).view(num_vecs, 1, 1)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    I = torch.eye(3).expand(num_vecs, 3, 3)
    
    R = I + sin_theta / theta * w_h + (1 - cos_theta) / (theta ** 2) * w_h2
    
    V = I + (1 - cos_theta) / (theta ** 2) * w_h + (theta - sin_theta) / (theta ** 3) * w_h2
    
    T = torch.zeros((num_vecs, 4, 4))
    T[:, 3, 3] = 1
    T[:, 0:3, 0:3] = R
    
    T[:, 0:3, 3:4] = torch.bmm(V, trans.unsqueeze(2))[:, :]
    return T
        


if __name__ == '__main__':
    # This file is not intended to be run directly. These are tests to make sure implemented functions are behaving as intended
    val = 2 * math.pi / (3 * math.sqrt(3))
    vec = torch.Tensor([-1, 0.5, 2, 1, 1, 1])
    T = lie_hat(vec)
    print(T)
    