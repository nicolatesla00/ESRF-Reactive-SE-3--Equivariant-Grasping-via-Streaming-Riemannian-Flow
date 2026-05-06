import torch
from copy import deepcopy
from torch.utils.checkpoint import checkpoint

from utils.Lie import inv_SO3, log_SO3, exp_so3, bracket_so3
from utils.ode_solvers import Lie_bracket


class EquiGraspFlow(torch.nn.Module):
    def __init__(self, p_uncond, guidance, init_dist, encoder, vector_field, ode_solver, 
                 sigma_0=0.01, k=0.0, streaming=False):
        super().__init__()

        self.p_uncond = p_uncond
        self.guidance = guidance
        self.sigma_0 = sigma_0  # Standard deviation for conditional distribution
        self.k = k  # Stabilizing gain
        self.streaming = streaming  # Enable streaming mode

        self.init_dist = init_dist
        self.encoder = encoder
        self.vector_field = vector_field
        self.ode_solver = ode_solver

    def step(self, data, losses, split, optimizer=None, use_conditional=False, last_pose=None):
        # Get data
        pc = data['pc']
        x_1 = data['Ts_grasp']

        # Get number of grasp poses in each batch and combine batched data
        nums_grasps = torch.tensor([len(Ts_grasp) for Ts_grasp in x_1], device=data['pc'].device)

        x_1 = torch.cat(x_1, dim=0)

        # Sample t and x_0
        t = torch.rand(len(x_1), 1).to(x_1.device)
        
        # Use conditional distribution if enabled and last_pose provided
        if use_conditional and last_pose is not None:
            # For training, use first pose of demonstration as "last pose"
            # In practice, this would be the last executed pose
            # Check if init_dist is a conditional distribution function
            if hasattr(self.init_dist, '__name__') and 'conditional' in self.init_dist.__name__:
                x_0 = self.init_dist(len(x_1), x_1.device, last_pose, self.sigma_0)
            else:
                # Fallback: manually create conditional distribution
                from utils.distributions import SE3_conditional_normal
                x_0 = SE3_conditional_normal(len(x_1), x_1.device, last_pose, self.sigma_0)
        else:
            x_0 = self.init_dist(len(x_1), x_1.device)

        # Get x_t and u_t (with stabilization if k > 0)
        if self.k > 0:
            x_t, u_t = get_traj_stabilized(x_0, x_1, t, k=self.k)
        else:
            x_t, u_t = get_traj(x_0, x_1, t)

        # Forward
        z_encoded = self.encoder(pc)
        z_for_mse = z_encoded.repeat_interleave(nums_grasps, dim=0)
        
        # Null condition for MSE 
        mask_uncond = torch.bernoulli(torch.Tensor([self.p_uncond] * len(z_for_mse))).to(bool)
        z_for_mse[mask_uncond] = torch.zeros_like(z_for_mse[mask_uncond])

        # Calculate MSE loss 
        v_t = self.vector_field(z_for_mse, t, x_t)
        loss_mse = losses['mse'](v_t, u_t)
        loss = losses['mse'].weight * loss_mse

        # Add geodesic consistency loss if available
        results = {
            f'scalar/{split}/loss': loss.item(),
            f'scalar/{split}/loss_mse': loss_mse.item(),
        }
        
        if 'geodesic_consistency' in losses:
            geodesic_loss = losses['geodesic_consistency']
            compute_frequency = getattr(geodesic_loss, 'compute_frequency', 0.2)
            
            
            should_compute = (split == 'val') or (torch.rand(1).item() < compute_frequency)
            
            if should_compute:

                z = z_encoded.repeat_interleave(nums_grasps, dim=0)
                
                traj = self._ode_solver_with_grad(z, x_0, self.guided_vector_field)
                

                x_1_hat = traj[:, -1]
                loss_geodesic = geodesic_loss(traj, x_0, x_1_hat)
                
                from utils.Lie import inv_SO3
                R_0 = x_0[:, :3, :3]
                R_1_hat = x_1_hat[:, :3, :3]
                
                actual_arc_lengths = geodesic_loss.calculate_trajectory_arc_length(traj)
                
                R_rel = torch.einsum('bij,bjk->bik', inv_SO3(R_0), R_1_hat)
                tr_R = torch.diagonal(R_rel, dim1=1, dim2=2).sum(1)
                theoretical_arc_lengths = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + 1e-6, 1 - 1e-6))
                
                arc_length_diff = (actual_arc_lengths - theoretical_arc_lengths).mean().item()
                
                loss = loss + geodesic_loss.weight * loss_geodesic
                
                results[f'scalar/{split}/loss'] = loss.item()
                results[f'scalar/{split}/loss_geodesic'] = geodesic_loss.weight * loss_geodesic.item()
                results[f'scalar/{split}/geodesic_error_rad'] = arc_length_diff
                results[f'scalar/{split}/geodesic_error_deg'] = arc_length_diff * 180 / 3.14159
                results[f'scalar/{split}/actual_arc_length_rad'] = actual_arc_lengths.mean().item()
                results[f'scalar/{split}/theoretical_arc_length_rad'] = theoretical_arc_lengths.mean().item()
            else:
                results[f'scalar/{split}/loss_geodesic'] = 0.0
                results[f'scalar/{split}/geodesic_error_rad'] = 0.0
                results[f'scalar/{split}/geodesic_error_deg'] = 0.0
                results[f'scalar/{split}/actual_arc_length_rad'] = 0.0
                results[f'scalar/{split}/theoretical_arc_length_rad'] = 0.0

        # Backward
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        return results

    def forward(self, pc, t, x_t, nums_grasps):
        z = torch.zeros((len(pc), self.encoder.dims[-1], 3), device=pc.device)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Null condition
        mask_uncond = torch.bernoulli(torch.Tensor([self.p_uncond] * len(z))).to(bool)

        z[mask_uncond] = torch.zeros_like(z[mask_uncond])

        # Get vector
        v_t = self.vector_field(z, t, x_t)

        return v_t

    @torch.no_grad()
    def sample(self, pc, nums_grasps, last_pose=None, geodesic_optimizer=None):

        # Sample initial samples (conditional if last_pose provided)
        if last_pose is not None:
            if hasattr(self.init_dist, '__name__') and 'conditional' in self.init_dist.__name__:
                x_0 = self.init_dist(sum(nums_grasps), pc.device, last_pose, self.sigma_0)
            else:
                from utils.distributions import SE3_conditional_normal
                x_0 = SE3_conditional_normal(sum(nums_grasps), pc.device, last_pose, self.sigma_0)
        else:
            x_0 = self.init_dist(sum(nums_grasps), pc.device)
        self.X0SAMPLED = deepcopy(x_0)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Get full trajectory for geodesic optimization if enabled
        if geodesic_optimizer is not None:
            # Get full trajectory
            traj = self.ode_solver(z, x_0, self.guided_vector_field)
            x_1_hat = traj[:, -1]
            
            # Optimize trajectory
            optimized_traj = geodesic_optimizer.optimize_trajectory(traj, x_0, x_1_hat)
            
            # Get optimized final poses
            x_1_hat = optimized_traj[:, -1]
        else:
            # Standard sampling without optimization
            x_1_hat = self.ode_solver(z, x_0, self.guided_vector_field)[:, -1]

        # Batch x_1_hat
        x_1_hat = x_1_hat.split(nums_grasps.tolist())

        return x_1_hat

    @torch.no_grad()
    def sample_streaming(self, pc, nums_grasps, last_pose=None, callback=None, 
                         pc_updater=None, check_interval=1, use_conditional_on_restart=True,
                         geodesic_optimizer=None):
        current_pc = pc
        current_last_pose = last_pose
        restart_count = 0
        max_restarts = 3  
        is_restart_after_update = False  
        
        # For geodesic optimization, we need to collect the full trajectory
        traj_list = [] if geodesic_optimizer is not None else None
        x_0_for_optimization = None
        
        while restart_count <= max_restarts:
            # Sample initial samples (conditional if last_pose provided and not restarting after update)
            # When object moves significantly, unconditional sampling may be better
            if current_last_pose is not None and (not is_restart_after_update or use_conditional_on_restart):
                if hasattr(self.init_dist, '__name__') and 'conditional' in self.init_dist.__name__:
                    x_0 = self.init_dist(sum(nums_grasps), current_pc.device, current_last_pose, self.sigma_0)
                else:
                    from utils.distributions import SE3_conditional_normal
                    x_0 = SE3_conditional_normal(sum(nums_grasps), current_pc.device, current_last_pose, self.sigma_0)
            else:
                x_0 = self.init_dist(sum(nums_grasps), current_pc.device)
            
            # Save x_0 for optimization if needed
            if geodesic_optimizer is not None and x_0_for_optimization is None:
                x_0_for_optimization = x_0.clone()
                traj_list = []

            # Encode point cloud
            z = self.encoder(current_pc)

            # Repeat feature
            z = z.repeat_interleave(nums_grasps, dim=0)

            # Check if solver supports streaming
            if hasattr(self.ode_solver, '_streaming_integrate'):
                # Streaming integration
                step = 0
                should_restart = False
                last_yielded_pose = None
                
                for x_t in self.ode_solver._streaming_integrate(z, x_0, self.guided_vector_field):
                    # Collect trajectory for optimization if needed
                    if geodesic_optimizer is not None:
                        traj_list.append(x_t.clone())
                    
                    # Check for point cloud updates
                    if pc_updater is not None and step % check_interval == 0:
                        new_pc = pc_updater()
                        if new_pc is not None and not torch.allclose(current_pc, new_pc, atol=1e-3):
                            # Point cloud updated, need to restart
                            # Optionally use current pose as last_pose for conditional sampling
                            # (controlled by use_conditional_on_restart parameter)
                            if use_conditional_on_restart:
                                current_last_pose = x_t[0:1]  # (1, 4, 4)
                            else:
                                current_last_pose = None  # Unconditional sampling for fresh start
                            current_pc = new_pc
                            is_restart_after_update = True
                            should_restart = True
                            # Reset trajectory collection on restart
                            if geodesic_optimizer is not None:
                                traj_list = []
                                x_0_for_optimization = None
                            if callback is not None:
                                # Notify callback about update
                                callback(x_t, step, should_update=True)
                            break
                    
                    # Call callback
                    if callback is not None:
                        should_continue = callback(x_t, step, should_update=False)
                        if should_continue is False:
                            should_restart = True
                            # Use current pose as last_pose for conditional sampling
                            # Take first grasp pose from batch
                            current_last_pose = x_t[0:1]  # (1, 4, 4)
                            # Reset trajectory collection on restart
                            if geodesic_optimizer is not None:
                                traj_list = []
                                x_0_for_optimization = None
                            break
                    
                    # Yield intermediate poses (before optimization)
                    yield x_t
                    last_yielded_pose = x_t
                    step += 1
                    
                    if should_restart:
                        break
                
                # If should restart, break inner loop and restart with new point cloud
                if should_restart:
                    restart_count += 1
                    if restart_count <= max_restarts:
                        continue
                    else:
                        # If max restarts reached, yield last pose (with optimization if enabled)
                        if last_yielded_pose is not None:
                            if geodesic_optimizer is not None and len(traj_list) > 0:
                                # Apply geodesic optimization to collected trajectory
                                traj = torch.stack(traj_list, dim=1)  # (N, num_steps, 4, 4)
                                x_1 = traj[:, -1]
                                optimized_traj = geodesic_optimizer.optimize_trajectory(traj, x_0_for_optimization, x_1)
                                yield optimized_traj[:, -1]
                            else:
                                yield last_yielded_pose
                        break
                else:
                    # Successfully completed - apply optimization if enabled
                    if geodesic_optimizer is not None and len(traj_list) > 0:
                        # Apply geodesic optimization to collected trajectory
                        traj = torch.stack(traj_list, dim=1)  # (N, num_steps, 4, 4)
                        x_1 = traj[:, -1]
                        optimized_traj = geodesic_optimizer.optimize_trajectory(traj, x_0_for_optimization, x_1)
                        # Yield optimized final pose
                        yield optimized_traj[:, -1]
                    is_restart_after_update = False  # Reset flag on successful completion
                    break
            else:
                # Fallback to batch mode
                x_1_hat = self.ode_solver(z, x_0, self.guided_vector_field)[:, -1]
                if geodesic_optimizer is not None:
                    # Get full trajectory for optimization
                    traj = self.ode_solver(z, x_0, self.guided_vector_field)
                    optimized_traj = geodesic_optimizer.optimize_trajectory(traj, x_0, x_1_hat)
                    x_1_hat = optimized_traj[:, -1]
                if callback is not None:
                    callback(x_1_hat, 0, should_update=False)
                yield x_1_hat
                break

    def guided_vector_field(self, z, t, x_t):
        v_t = (1 - self.guidance) * self.vector_field(torch.zeros_like(z), t, x_t) + self.guidance * self.vector_field(z, t, x_t)

        return v_t


def get_traj(x_0, x_1, t):
    # Get rotations
    R_0 = x_0[:, :3, :3]
    R_1 = x_1[:, :3, :3]

    # Get translations
    p_0 = x_0[:, :3, 3]
    p_1 = x_1[:, :3, 3]

    # Get x_t
    x_t = torch.eye(4).repeat(len(x_1), 1, 1).to(x_1)
    x_t[:, :3, :3] = (R_0 @ exp_so3(t.unsqueeze(2) * log_SO3(inv_SO3(R_0) @ R_1)))
    x_t[:, :3, 3] = p_0 + t * (p_1 - p_0)

    # Get u_t
    u_t = torch.zeros(len(x_1), 6).to(x_1)
    u_t[:, :3] = bracket_so3(log_SO3(inv_SO3(R_0) @ R_1))
    u_t[:, :3] = torch.einsum('bij,bj->bi', R_0, u_t[:, :3])    # Convert w_b to w_s
    u_t[:, 3:] = p_1 - p_0

    return x_t, u_t


def get_traj_stabilized(x_0, x_1, t, k=0.0):

    # Get standard trajectory
    x_t, u_t = get_traj(x_0, x_1, t)
    
    if k <= 0:
        return x_t, u_t
    
    # Compute SE(3) error for stabilization
    R_t = x_t[:, :3, :3]
    R_1 = x_1[:, :3, :3]
    p_t = x_t[:, :3, 3]
    p_1 = x_1[:, :3, 3]
    
    # Rotation error in so(3) Lie algebra
    R_error = log_SO3(inv_SO3(R_t) @ R_1)  # (B, 3, 3)
    R_error_vec = bracket_so3(R_error)  # (B, 3)
    
    # Convert to spatial frame (w_s)
    R_error_vec = torch.einsum('bij,bj->bi', R_t, R_error_vec)
    
    # Translation error
    p_error = p_1 - p_t  # (B, 3)
    
    # Stabilized velocity: subtract stabilization term
    u_stabilized = u_t.clone()
    u_stabilized[:, :3] -= k * R_error_vec
    u_stabilized[:, 3:] -= k * p_error
    
    return x_t, u_stabilized


def _ode_step_with_grad(self, x_n, z, t_n, h, func):

    ##### Stage 1 #####
    x_hat_1 = x_n
    V_1 = func(z, t_n, x_hat_1)
    w_1 = V_1[:, :3]
    v_1 = V_1[:, 3:]
    w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
    w_1 = bracket_so3(w_1)
    I_1 = w_1

    ##### Stage 2 #####
    u_2 = h.unsqueeze(-1) * (1 / 2) * w_1
    u_2 = u_2 + (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_2)
    R_2 = x_n[:, :3, :3] @ exp_so3(u_2)
    p_2 = x_n[:, :3, 3] + h * (v_1 / 2)
    row_3 = torch.cat([
        torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
        torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
    ], dim=-1)
    x_hat_2 = torch.cat([
        torch.cat([R_2, p_2.unsqueeze(-1)], dim=-1),
        row_3
    ], dim=1)
    V_2 = func(z, t_n + (h / 2), x_hat_2)
    w_2 = V_2[:, :3]
    v_2 = V_2[:, 3:]
    w_2 = torch.einsum('bji,bj->bi', x_hat_2[:, :3, :3], w_2)
    w_2 = bracket_so3(w_2)

    ##### Stage 3 #####
    u_3 = h.unsqueeze(-1) * (1 / 2) * w_2
    u_3 = u_3 + (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_3)
    R_3 = x_n[:, :3, :3] @ exp_so3(u_3)
    p_3 = x_n[:, :3, 3] + h * (v_2 / 2)
    row_3 = torch.cat([
        torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
        torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
    ], dim=-1)
    x_hat_3 = torch.cat([
        torch.cat([R_3, p_3.unsqueeze(-1)], dim=-1),
        row_3
    ], dim=1)
    V_3 = func(z, t_n + (h / 2), x_hat_3)
    w_3 = V_3[:, :3]
    v_3 = V_3[:, 3:]
    w_3 = torch.einsum('bji,bj->bi', x_hat_3[:, :3, :3], w_3)
    w_3 = bracket_so3(w_3)

    ##### Stage 4 #####
    u_4 = h.unsqueeze(-1) * w_3
    u_4 = u_4 + (h.unsqueeze(-1) / 6) * Lie_bracket(I_1, u_4)
    R_4 = x_n[:, :3, :3] @ exp_so3(u_4)
    p_4 = x_n[:, :3, 3] + h * v_3
    row_3 = torch.cat([
        torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
        torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
    ], dim=-1)
    x_hat_4 = torch.cat([
        torch.cat([R_4, p_4.unsqueeze(-1)], dim=-1),
        row_3
    ], dim=1)
    V_4 = func(z, t_n + h, x_hat_4)
    w_4 = V_4[:, :3]
    v_4 = V_4[:, 3:]
    w_4 = torch.einsum('bji,bj->bi', x_hat_4[:, :3, :3], w_4)
    w_4 = bracket_so3(w_4)

    ##### Update #####
    I_2 = (2 * (w_2 - I_1) + 2 * (w_3 - I_1) - (w_4 - I_1)) / h.unsqueeze(-1)
    u = h.unsqueeze(-1) * (1 / 6 * w_1 + 1 / 3 * w_2 + 1 / 3 * w_3 + 1 / 6 * w_4)
    u = u + (h.unsqueeze(-1) / 4) * Lie_bracket(I_1, u) + ((h ** 2).unsqueeze(-1) / 24) * Lie_bracket(I_2, u)

    R_next = x_n[:, :3, :3] @ exp_so3(u)
    p_next = x_n[:, :3, 3] + (h / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4)
    row_3 = torch.cat([
        torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
        torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
    ], dim=-1)
    x_next = torch.cat([
        torch.cat([R_next, p_next.unsqueeze(-1)], dim=-1),
        row_3
    ], dim=1)
    
    return x_next


def _ode_solver_with_grad(self, z, x_0, func):

    num_steps = len(self.ode_solver.t) - 1
    t = self.ode_solver.t.to(z.device)
    dt = t[1:] - t[:-1]
    
    traj_list = [x_0]
    
    guidance = self.guidance
    vector_field = self.vector_field
    
    def guided_vector_field_wrapper(z, t, x_t):
        return (1 - guidance) * vector_field(torch.zeros_like(z), t, x_t) + guidance * vector_field(z, t, x_t)
    
    def _ode_step_wrapper(x_n, z, t_n, h):
        ##### Stage 1 #####
        x_hat_1 = x_n
        V_1 = guided_vector_field_wrapper(z, t_n, x_hat_1)
        w_1 = V_1[:, :3]
        v_1 = V_1[:, 3:]
        w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
        w_1 = bracket_so3(w_1)
        I_1 = w_1

        ##### Stage 2 #####
        u_2 = h.unsqueeze(-1) * (1 / 2) * w_1
        u_2 = u_2 + (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_2)
        R_2 = x_n[:, :3, :3] @ exp_so3(u_2)
        p_2 = x_n[:, :3, 3] + h * (v_1 / 2)
        row_3 = torch.cat([
            torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
            torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
        ], dim=-1)
        x_hat_2 = torch.cat([
            torch.cat([R_2, p_2.unsqueeze(-1)], dim=-1),
            row_3
        ], dim=1)
        V_2 = guided_vector_field_wrapper(z, t_n + (h / 2), x_hat_2)
        w_2 = V_2[:, :3]
        v_2 = V_2[:, 3:]
        w_2 = torch.einsum('bji,bj->bi', x_hat_2[:, :3, :3], w_2)
        w_2 = bracket_so3(w_2)

        ##### Stage 3 #####
        u_3 = h.unsqueeze(-1) * (1 / 2) * w_2
        u_3 = u_3 + (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_3)
        R_3 = x_n[:, :3, :3] @ exp_so3(u_3)
        p_3 = x_n[:, :3, 3] + h * (v_2 / 2)
        row_3 = torch.cat([
            torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
            torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
        ], dim=-1)
        x_hat_3 = torch.cat([
            torch.cat([R_3, p_3.unsqueeze(-1)], dim=-1),
            row_3
        ], dim=1)
        V_3 = guided_vector_field_wrapper(z, t_n + (h / 2), x_hat_3)
        w_3 = V_3[:, :3]
        v_3 = V_3[:, 3:]
        w_3 = torch.einsum('bji,bj->bi', x_hat_3[:, :3, :3], w_3)
        w_3 = bracket_so3(w_3)

        ##### Stage 4 #####
        u_4 = h.unsqueeze(-1) * w_3
        u_4 = u_4 + (h.unsqueeze(-1) / 6) * Lie_bracket(I_1, u_4)
        R_4 = x_n[:, :3, :3] @ exp_so3(u_4)
        p_4 = x_n[:, :3, 3] + h * v_3
        row_3 = torch.cat([
            torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
            torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
        ], dim=-1)
        x_hat_4 = torch.cat([
            torch.cat([R_4, p_4.unsqueeze(-1)], dim=-1),
            row_3
        ], dim=1)
        V_4 = guided_vector_field_wrapper(z, t_n + h, x_hat_4)
        w_4 = V_4[:, :3]
        v_4 = V_4[:, 3:]
        w_4 = torch.einsum('bji,bj->bi', x_hat_4[:, :3, :3], w_4)
        w_4 = bracket_so3(w_4)

        ##### Update #####
        I_2 = (2 * (w_2 - I_1) + 2 * (w_3 - I_1) - (w_4 - I_1)) / h.unsqueeze(-1)
        u = h.unsqueeze(-1) * (1 / 6 * w_1 + 1 / 3 * w_2 + 1 / 3 * w_3 + 1 / 6 * w_4)
        u = u + (h.unsqueeze(-1) / 4) * Lie_bracket(I_1, u) + ((h ** 2).unsqueeze(-1) / 24) * Lie_bracket(I_2, u)

        R_next = x_n[:, :3, :3] @ exp_so3(u)
        p_next = x_n[:, :3, 3] + (h / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4)
        row_3 = torch.cat([
            torch.zeros(len(x_n), 1, 3, device=x_n.device, dtype=x_n.dtype),
            torch.ones(len(x_n), 1, 1, device=x_n.device, dtype=x_n.dtype)
        ], dim=-1)
        x_next = torch.cat([
            torch.cat([R_next, p_next.unsqueeze(-1)], dim=-1),
            row_3
        ], dim=1)
        
        return x_next
    
    checkpoint_every = 1
    
    for n in range(num_steps):
        x_n = traj_list[n].contiguous()
        t_n = t[n].repeat(len(x_0), 1)
        h = dt[n].repeat(len(x_0), 1)
        
        if n % checkpoint_every == 0:
            x_next = checkpoint(
                _ode_step_wrapper,
                x_n, z, t_n, h,
                use_reentrant=False
            )
        else:
            x_next = self._ode_step_with_grad(x_n, z, t_n, h, guided_vector_field_wrapper)
        
        traj_list.append(x_next)
    
    traj = torch.stack(traj_list, dim=1)
    return traj


EquiGraspFlow._ode_step_with_grad = _ode_step_with_grad
EquiGraspFlow._ode_solver_with_grad = _ode_solver_with_grad
