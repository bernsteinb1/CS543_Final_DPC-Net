"""
Bryce Bernstein
---------------
This file is a wrapper for lie algebra functions within DPC-Net.
I used https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
and https://natanaso.github.io/ece276a2020/ref/ECE276A_12_SO3_SE3.pdf for help understanding matrix
exponentials and logarithms for developing this file.
"""
import torch
import math


def so3_wedge(rotations: torch.Tensor):
    if rotations.dim() == 1:
        rotations = rotations.unsqueeze(0)
    matrices = torch.zeros((rotations.size(0), 3, 3), device=rotations.device, dtype=rotations.dtype)
    matrices[:, 0, 1] = -rotations[:, 2]
    matrices[:, 0, 2] = rotations[:, 1]
    matrices[:, 1, 0] = rotations[:, 2]
    matrices[:, 1, 2] = -rotations[:, 0]
    matrices[:, 2, 0] = -rotations[:, 1]
    matrices[:, 2, 1] = rotations[:, 0]
    return matrices

def so3_vee(wedge: torch.Tensor):
    if wedge.dim() == 2:
        wedge = wedge.unsqueeze(0)
    return wedge[:, [2, 0, 1], [1, 2, 0]]

def so3_exp(rots: torch.Tensor):
    if rots.dim() == 1:
        rots = rots.unsqueeze(0)
    num_vecs = rots.size(0)
    t_h = so3_wedge(rots)
    t_h2 = t_h @ t_h
    
    theta = torch.linalg.vector_norm(rots, dim=1).view(num_vecs, 1, 1)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    I = torch.eye(3, dtype=rots.dtype, device=rots.device).expand(num_vecs, 3, 3)
    
    R = I + sin_theta / theta * t_h + (1 - cos_theta) / (theta ** 2) * t_h2
    return R

def so3_log(rots: torch.Tensor):
    if rots.dim() == 2:
        rots = rots.unsqueeze(0)
    num_vecs = rots.size(0)
    traces = rots.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1)
    theta = torch.arccos((traces - 1) / 2).view((num_vecs, 1, 1))
    t_w = theta / (2 * torch.sin(theta)) * (rots - rots.transpose(1, 2))
    return so3_vee(t_w)

def so3_l_jacobian(rots: torch.Tensor):
    if rots.dim() == 1:
        rots = rots.unsqueeze(0)
    n = rots.size(0)
        
    theta = torch.linalg.vector_norm(rots, dim=1).view(n, 1, 1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    t_h = so3_wedge(rots)
    t_h2 = t_h @ t_h
    
    I = torch.eye(3, dtype=rots.dtype, device=rots.device).expand(n, 3, 3)
    J = I + (1 - cos_theta) / theta ** 2 * t_h + (theta - sin_theta) / theta ** 3 * t_h2
    return J

def so3_l_jacobian_inv(rots: torch.Tensor):
    if rots.dim() == 1:
        rots = rots.unsqueeze(0)
    n = rots.size(0)
        
    theta = torch.linalg.vector_norm(rots, dim=1).view(n, 1, 1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    t_h = so3_wedge(rots)
    t_h2 = t_h @ t_h
    
    I = torch.eye(3, dtype=rots.dtype, device=rots.device).expand(n, 3, 3)
    J_inv = I - 1 / 2 * t_h + (1 / theta ** 2 - (1 + cos_theta) / (2 * theta * sin_theta)) * t_h2
    return J_inv

def se3_wedge(se3_trans: torch.Tensor):
    if se3_trans.dim() == 1:
        se3_trans = se3_trans.unsqueeze(0)
    num_trans = se3_trans.size(0)
    T = torch.zeros((num_trans, 4, 4), device=se3_trans.device, dtype=se3_trans.dtype)
    T[:, 0:3, 0:3] = so3_wedge(se3_trans[:, 3:])
    T[:, 0:3, 3] = se3_trans[:, :3]
    
    return T

def se3_curlywedge(se3_trans: torch.Tensor):
    if se3_trans.dim() == 1:
        se3_trans = se3_trans.unsqueeze(0)
    n = se3_trans.size(0)
    theta = so3_wedge(se3_trans[:, 3:])
    rho = so3_wedge(se3_trans[:, :3])
    
    cw = torch.zeros((n, 6, 6), dtype=se3_trans.dtype, device=se3_trans.device)
    cw[:, 0:3, 0:3] = theta
    cw[:, 3:6, 3:6] = theta
    cw[:, 0:3, 3:6] = rho
    
    return cw

def se3_l_jacobian(trans: torch.Tensor):
    if trans.dim() == 1:
        trans = trans.unsqueeze(0)
    n = trans.size(0)
    
    e_cw1 = se3_curlywedge(trans)
    e_cw2 = e_cw1 @ e_cw1
    e_cw3 = e_cw2 @ e_cw1
    e_cw4 = e_cw3 @ e_cw1
    
    theta = torch.linalg.vector_norm(trans[:, 3:], dim=1).view(n, 1, 1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    coeff1 = (4 - theta * sin_theta - 4 * cos_theta) / (2 * theta ** 2)
    coeff2 = (4 * theta - 5 * sin_theta + theta * cos_theta) / (2 * theta ** 3)
    coeff3 = (2 - theta * sin_theta - 2 * cos_theta) / (2 * theta ** 4)
    coeff4 = (2 * theta - 3 * sin_theta + theta * cos_theta) / (2 * theta ** 5)
    
    I = torch.eye(6, dtype=trans.dtype, device=trans.device)
    
    Jl = I + coeff1 * e_cw1 + coeff2 * e_cw2 + coeff3 * e_cw3 + coeff4 * e_cw4
    return Jl

def se3_l_jacobian_inv(trans: torch.Tensor):
    Jl = se3_l_jacobian(trans)
    J = Jl[:, 0:3, 0:3]
    J_inv = torch.linalg.inv(J)
    Q = Jl[:, 0:3, 3:6]
    
    inverse = torch.zeros_like(Jl)
    inverse[:, 0:3, 0:3] = J_inv
    inverse[:, 3:6, 3:6] = J_inv
    inverse[:, 0:3, 3:6] = -J_inv @ Q @ J_inv
    return inverse

def se3_log(matrices: torch.Tensor):
    if matrices.dim() == 2:
        matrices = matrices.unsqueeze(0)
    n = matrices.size(0)
    rots = matrices[:, 0:3, 0:3]
    
    theta = so3_log(rots)
    p = matrices[:, 0:3, 3:]
    Jl_inv = so3_l_jacobian_inv(so3_log(rots))
    rho = torch.bmm(Jl_inv, p).squeeze(2)
    
    xi = torch.cat((rho, theta), 1)
    return xi
    

def se3_exp(vectors: torch.Tensor):
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)
    num_vecs = vectors.size(0)
    e_h = se3_wedge(vectors)
    e_h2 = e_h @ e_h
    e_h3 = e_h2 @ e_h
    
    theta = torch.linalg.vector_norm(vectors[:, 3:], dim=1).view(num_vecs, 1, 1)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    I = torch.eye(4, dtype=vectors.dtype, device=vectors.device).expand(num_vecs, 4, 4)
    
    T = I + e_h + (1 - cos_theta) / (theta ** 2) * e_h2 + (theta - sin_theta) / theta ** 3 * e_h3
    
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
    T = se3_exp(vec)
    print(se3_log(T))
    # print(T)
    