import torch
from utils.Lie import inv_SO3, log_SO3, bracket_so3, exp_so3


class GeodesicConsistencyLoss:
   
    def __init__(self, weight=1.0, reduction='mean', compute_frequency=0.2):
        self.weight = weight
        self.reduction = reduction
        self.compute_frequency = compute_frequency
    
    def calculate_SO3_geodesic_distance(self, R_0, R_1):

        R_rel = torch.einsum('bij,bjk->bik', inv_SO3(R_0), R_1)
        
        tr_R = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
        theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + 1e-6, 1 - 1e-6))
        
        return theta
    
    def calculate_trajectory_arc_length(self, traj):
        
        N, num_points, _, _ = traj.shape
        arc_lengths = torch.zeros(N, device=traj.device)
        
        R_traj = traj[:, :, :3, :3]  # (N, num_points, 3, 3)
        
        for i in range(num_points - 1):
            R_i = R_traj[:, i]
            R_next = R_traj[:, i + 1]
            distances = self.calculate_SO3_geodesic_distance(R_i, R_next)
            arc_lengths += distances
        
        return arc_lengths
    
    def __call__(self, traj, x_0, x_1):
        
        R_0 = x_0[:, :3, :3]  # (N, 3, 3)
        R_1 = x_1[:, :3, :3]  # (N, 3, 3)
        
        actual_arc_lengths = self.calculate_trajectory_arc_length(traj)
        
        theoretical_arc_lengths = self.calculate_SO3_geodesic_distance(R_0, R_1)
        

        diff = actual_arc_lengths - theoretical_arc_lengths
        loss = diff ** 2
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class GeodesicSmoothnessLoss:

    def __init__(self, weight=1.0, reduction='mean'):
        self.weight = weight
        self.reduction = reduction
    
    def calculate_SO3_geodesic_distance(self, R_0, R_1):
        R_rel = torch.einsum('bij,bjk->bik', inv_SO3(R_0), R_1)
        tr_R = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
        theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + 1e-6, 1 - 1e-6))
        return theta
    
    def __call__(self, traj):
       
        N, num_points, _, _ = traj.shape
        R_traj = traj[:, :, :3, :3]  # (N, num_points, 3, 3)
        
        distances = []
        for i in range(num_points - 1):
            R_i = R_traj[:, i]
            R_next = R_traj[:, i + 1]
            dist = self.calculate_SO3_geodesic_distance(R_i, R_next)
            distances.append(dist)
        
        if len(distances) < 2:
            return torch.tensor(0.0, device=traj.device)
        
        distances = torch.stack(distances, dim=1)  # (N, num_points-1)
        
        first_diff = distances[:, 1:] - distances[:, :-1]  # (N, num_points-2)
        
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]  # (N, num_points-3)
        
        loss = second_diff ** 2
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

