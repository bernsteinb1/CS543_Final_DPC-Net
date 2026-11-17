"""
Bryce Bernstein
---------------
This file is a wrapper for lie algebra functions within DPC-Net
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
    if vectors.size(0) == 6:
        vectors = vectors.unsqueeze(0)   
    num_vecs = vectors.size(0)
    
    trans = vectors[:, :3]
    theta = torch.linalg.vector_norm(vectors[:, 3:], dim=1, keepdim=True)
    rots = vectors[:, 3:] / theta
    w_h = skew_matrix(rots)
    w_h2 = w_h @ w_h
    
    sin_theta = torch.sin(theta).reshape(num_vecs, 1, 1)
    cos_theta = torch.cos(theta).reshape(num_vecs, 1, 1)
    
    I = torch.eye(3).expand(num_vecs, 3, 3)
    
    R = I + w_h * sin_theta + w_h2 * (1 - cos_theta)
    print(R)
    print(torch.bmm(R, R.transpose(1, 2)))
        


if __name__ == '__main__':
    # This file is not intended to be run directly. These are tests to make sure implemented functions are behaving as intended
    val = 2 * math.pi / (3 * math.sqrt(3))
    vec = torch.Tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, val, val, val], [1, 1, 1, 3 * math.pi, 0, 0]])
    lie_hat(vec)
    pass