import roma
import torch

from utils.Lie import exp_so3, bracket_so3


def get_dist(cfg):
    name = cfg.pop('name')

    if name == 'SO3_uniform_R3_normal':
        dist_fn = SO3_uniform_R3_normal
    elif name == 'SO3_uniform_R3_spherical':
        dist_fn = SO3_uniform_R3_spherical
    elif name == 'SO3_centripetal_R3_normal':
        dist_fn = SO3_centripetal_R3_normal
    elif name == 'SO3_centripetal_R3_spherical':
        dist_fn = SO3_centripetal_R3_spherical
    elif name == 'SE3_conditional_normal':
        dist_fn = SE3_conditional_normal
    elif name == 'SE3_conditional_spherical':
        dist_fn = SE3_conditional_spherical
    else:
        raise NotImplementedError(f"Distribution {name} is not implemented.")
    
    return dist_fn


def SO3_uniform_R3_normal(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = torch.randn(num_samples, 3).to(device)

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_uniform_R3_spherical(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = torch.randn(num_samples, 3).to(device)
    p /= p.norm(dim=-1, keepdim=True)

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_centripetal_R3_normal(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = - (0.112 * 5 + torch.randn(num_samples, 1).to(device).abs()) * R[:, :, 2]

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SO3_centripetal_R3_spherical(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = - R[:, :, 2]

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    return T


def SE3_conditional_normal(num_samples, device, last_pose=None, sigma_0=0.01):

    if last_pose is None:
        # Fallback to standard distribution
        return SO3_uniform_R3_normal(num_samples, device)
    
    # Handle both single pose and batch of poses
    if last_pose.dim() == 2:
        last_pose = last_pose.unsqueeze(0).repeat(num_samples, 1, 1)
    elif last_pose.dim() == 3:
        if len(last_pose) == 1:
            last_pose = last_pose.repeat(num_samples, 1, 1)
        elif len(last_pose) != num_samples:
            raise ValueError(f"last_pose batch size {len(last_pose)} != num_samples {num_samples}")
    
    last_pose = last_pose.to(device)
    
    # Extract rotation and translation
    R_last = last_pose[:, :3, :3]  # (N, 3, 3)
    p_last = last_pose[:, :3, 3]   # (N, 3)
    
    # Add noise to rotation (in so(3) Lie algebra)
    noise_R = torch.randn(num_samples, 3).to(device) * sigma_0  # (N, 3)
    R_noise = R_last @ exp_so3(noise_R)  # (N, 3, 3)
    
    # Add noise to translation (in R^3)
    p_noise = p_last + torch.randn(num_samples, 3).to(device) * sigma_0  # (N, 3)
    
    # Combine
    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R_noise
    T[:, :3, 3] = p_noise
    
    return T


def SE3_conditional_spherical(num_samples, device, last_pose=None, sigma_0=0.01):

    if last_pose is None:
        # Fallback to standard distribution
        return SO3_uniform_R3_spherical(num_samples, device)
    
    # Handle both single pose and batch of poses
    if last_pose.dim() == 2:
        last_pose = last_pose.unsqueeze(0).repeat(num_samples, 1, 1)
    elif last_pose.dim() == 3:
        if len(last_pose) == 1:
            last_pose = last_pose.repeat(num_samples, 1, 1)
        elif len(last_pose) != num_samples:
            raise ValueError(f"last_pose batch size {len(last_pose)} != num_samples {num_samples}")
    
    last_pose = last_pose.to(device)
    
    # Extract rotation and translation
    R_last = last_pose[:, :3, :3]  # (N, 3, 3)
    p_last = last_pose[:, :3, 3]   # (N, 3)
    
    # Add noise to rotation (in so(3) Lie algebra)
    noise_R = torch.randn(num_samples, 3).to(device) * sigma_0  # (N, 3)
    R_noise = R_last @ exp_so3(noise_R)  # (N, 3, 3)
    
    # Add noise to translation and normalize (spherical)
    p_noise = p_last + torch.randn(num_samples, 3).to(device) * sigma_0  # (N, 3)
    p_noise = p_noise / (p_noise.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize
    
    # Combine
    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R_noise
    T[:, :3, 3] = p_noise

    return T
