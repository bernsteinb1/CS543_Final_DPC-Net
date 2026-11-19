"""
Bryce Bernstein
---------------
This file is a wrapper for lie algebra functions within DPC-Net.
I used https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf for help understanding matrix
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

def lie_vee(matrices: torch.Tensor):
    if matrices.dim() == 2:
        matrices = matrices.unsqueeze(0)
    num_vecs = matrices.size(0)
    rots = matrices[:, 0:3, 0:3]
    traces = rots.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1)
    vecs = (rots - rots.transpose(1, 2)).flatten(1)[:, [7, 2, 3]]
    
    cos_theta = (1 / 2 * (traces - 1)).view(num_vecs, 1)
    sin_theta = (1 / 2 * torch.sqrt((3 - traces) * (1 + traces))).view(num_vecs, 1)
    theta = torch.atan2(sin_theta, cos_theta)
    w = theta / (2 * sin_theta) * (vecs)
    w_h = skew_matrix(w)
    w_h2 = w_h @ w_h
    
    cos_theta_over_2 = torch.cos(theta / 2)
    sin_theta_over_2 = torch.sin(theta / 2)
    V_inv = torch.eye(3).expand(num_vecs, 3, 3) - 1 / 2 * w_h + ((1 - (theta * cos_theta_over_2 / (2 * sin_theta_over_2))) / theta ** 2).view(num_vecs, 1, 1) * w_h2
    
    trans = matrices[:, 0:3, 3:4]
    lie_algebra = torch.cat((torch.bmm(V_inv, trans).squeeze(2), w), dim=1)
    
    return lie_algebra
    

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
    vec = torch.Tensor([
        [1, 2, 3, 0.1, 0.2, 0.3],
        [-0.5, 0.25, -1.5, 0.1, -0.2, 0.3],
        [43, 2, 4, 1, 0, 0.5],
        [.75, 1.25, -2, 0.05, 0.35, -.2],
        [2, -1, .5, -.5, .25, .15]
        ])
    T = lie_hat(vec)
    print(lie_vee(T))
    # print(T)
    