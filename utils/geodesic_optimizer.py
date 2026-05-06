import torch
from utils.Lie import inv_SO3, log_SO3, exp_so3, bracket_so3


class GeodesicOptimizer:

    def __init__(self, num_iterations=10, step_size=0.1, tolerance=1e-6, use_simple_interpolation=False):

        self.num_iterations = num_iterations
        self.step_size = step_size
        self.tolerance = tolerance
        self.use_simple_interpolation = use_simple_interpolation
    
    def calculate_SO3_geodesic_distance(self, R_0, R_1):
        R_rel = torch.einsum('bij,bjk->bik', inv_SO3(R_0), R_1)
        tr_R = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
        theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + 1e-6, 1 - 1e-6))
        return theta
    
    def calculate_trajectory_arc_length(self, traj):
        N, num_points, _, _ = traj.shape
        arc_lengths = torch.zeros(N, device=traj.device)
        R_traj = traj[:, :, :3, :3]
        
        for i in range(num_points - 1):
            R_i = R_traj[:, i]
            R_next = R_traj[:, i + 1]
            distances = self.calculate_SO3_geodesic_distance(R_i, R_next)
            arc_lengths += distances
        
        return arc_lengths
    
    def optimize_trajectory(self, traj, x_0, x_1):
        traj = traj.clone()
        R_0 = x_0[:, :3, :3]
        R_1 = x_1[:, :3, :3]
        
        theoretical_arc_length = self.calculate_SO3_geodesic_distance(R_0, R_1)
        
        N, num_points, _, _ = traj.shape
        t_values = torch.linspace(0, 1, num_points, device=traj.device)
        
        R_rel = torch.einsum('bij,bjk->bik', inv_SO3(R_0), R_1)
        log_R_rel = log_SO3(R_rel)
        w_vec = bracket_so3(log_R_rel)  # (N, 3)
        
        R_geodesic = []
        for t_val in t_values:
            w_vec_t = t_val * w_vec
            R_t = R_0 @ exp_so3(w_vec_t)
            R_geodesic.append(R_t)
        R_geodesic = torch.stack(R_geodesic, dim=1)  # (N, num_points, 3, 3)
        
        if self.use_simple_interpolation:
            traj[:, :, :3, :3] = R_geodesic
            p_0 = x_0[:, :3, 3]
            p_1 = x_1[:, :3, 3]
            for i in range(num_points):
                t = t_values[i]
                traj[:, i, :3, 3] = p_0 + t * (p_1 - p_0)
            return traj
        

        original_traj = traj.clone()
        best_traj = traj.clone()
        best_error = float('inf')
        
        initial_arc_length = self.calculate_trajectory_arc_length(traj)
        initial_error = torch.abs(initial_arc_length - theoretical_arc_length).mean().item()
        best_error = initial_error
        
        for iteration in range(self.num_iterations):
            R_traj = traj[:, :, :3, :3]
            
            current_arc_length = self.calculate_trajectory_arc_length(traj)
            error = torch.abs(current_arc_length - theoretical_arc_length)
            mean_error = error.mean().item()
            
            if mean_error < best_error:
                best_error = mean_error
                best_traj = traj.clone()
            
            if mean_error < self.tolerance:
                break
            

            improved = False
            

            interpolation_weight = min(self.step_size * 50, mean_error * 2.0)
            interpolation_weight = min(interpolation_weight, 0.3)  
            interpolation_weight = max(interpolation_weight, 0.01)  
            
            R_candidates = R_traj.clone()
            
            for i in range(1, num_points - 1):  
                R_current = R_traj[:, i]  
                R_geodesic_i = R_geodesic[:, i]  
                
                R_rel_step = torch.einsum('bij,bjk->bik', inv_SO3(R_current), R_geodesic_i)
                log_R_rel_step = log_SO3(R_rel_step)
                w_vec_step = bracket_so3(log_R_rel_step)  # (N, 3)
                
                step_norm = w_vec_step.norm(dim=1)  # (N,)
                
                mask = step_norm > 1e-6
                if not mask.any():
                    continue
                
                move_distance = interpolation_weight * step_norm
                move_distance = torch.minimum(move_distance, step_norm * 0.3)  
                
                w_vec_step_normalized = w_vec_step / (step_norm.unsqueeze(1) + 1e-8)  # (N, 3)
                
                step = move_distance.unsqueeze(1) * w_vec_step_normalized  # (N, 3)
                R_updated = R_current.clone()
                R_updated[mask] = R_current[mask] @ exp_so3(step[mask])
                R_candidates[:, i] = R_updated
            
            traj_candidate = traj.clone()
            traj_candidate[:, 1:num_points-1, :3, :3] = R_candidates[:, 1:num_points-1]
            
            candidate_arc_length = self.calculate_trajectory_arc_length(traj_candidate)
            candidate_error = torch.abs(candidate_arc_length - theoretical_arc_length).mean().item()
            
            if candidate_error < mean_error:
                traj = traj_candidate
                improved = True
            
            p_0 = x_0[:, :3, 3]
            p_1 = x_1[:, :3, 3]
            for i in range(num_points):
                t = t_values[i]
                traj[:, i, :3, 3] = p_0 + t * (p_1 - p_0)
        
        final_arc_length = self.calculate_trajectory_arc_length(best_traj)
        final_error = torch.abs(final_arc_length - theoretical_arc_length).mean().item()
        
        if final_error > initial_error:
            return original_traj
        
        return best_traj
    
    @torch.no_grad()
    def optimize_batch(self, model, pc, nums_grasps, x_0=None):
        
        model.eval()
        
        if x_0 is None:
            x_0 = model.init_dist(sum(nums_grasps), pc.device)
        
        z = model.encoder(pc)
        z = z.repeat_interleave(nums_grasps, dim=0)
        
        traj = model.ode_solver(z, x_0, model.guided_vector_field)
        
        x_1 = traj[:, -1]
        
        optimized_traj = self.optimize_trajectory(traj, x_0, x_1)
        
        optimized_trajs = optimized_traj.split(nums_grasps.tolist())
        
        return optimized_trajs

