import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.distributions import Normal
import torch.distributed as dist
import math
import numpy as np
import torch.distributed as dist    
from tqdm.auto import tqdm
import json
import torch.nn.functional as F
from einops import rearrange, reduce
from helpers.util import preprocess_angle2sincos,descale_box_params,postprocess_sincos2arctan
from .loss import axis_aligned_bbox_overlaps_3d
from .oriented_iou_loss import cal_iou_3d
from .physical_guidance import (
    compute_room_outer_loss, 
    compute_walkable_loss, 
    compute_center_penalty_loss, 
    compute_pathfinding_walkable_loss, 
    compute_edge_gaussian_walkable_loss,
    compute_relational_guidance_loss
)
#from helpers.threedfront_box3d import bbox_overlaps_3d, axis_aligned_bbox_overlaps_3d


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm

def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'cosine':

        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            """
            Create a beta schedule that discretizes the given alpha_t_bar function,
            which defines the cumulative product of (1-beta) over time from t = [0,1].
            :param num_diffusion_timesteps: the number of betas to produce.
            :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
            :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
            """
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            
            return np.array(betas).astype(np.float64)
        
        betas_for_alpha_bar(
            time_num,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(schedule_type)
    return betas

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self, config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou, iou_type, train_stats_file):
        # read object property dimension
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.bbox_norm_file = train_stats_file
        self.box_stats = np.loadtxt(train_stats_file).astype(np.float32) if train_stats_file is not None else None
        self.loss_separate = loss_separate
        self.loss_iou = loss_iou
        self.iou_type = iou_type
        self.inference_guidance = cfg_get(config, "inference_guidance", None)
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # borrowed from SDFusion
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        self.latest_sampling_stats = None

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):  
        """
        diffusion step: q(x_t | x_{t-1})
        """
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)   q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, obj_embed, triples, condition, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, obj_embed, triples, t, condition)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0) 

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        
        elif self.model_mean_type == 'x0':
            x_recon = model_output

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0) 

            eps = self._predict_eps_from_start(data, t=t, x0=x_recon)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )
    
    def _predict_eps_from_start(self, x_t, t, x0):
        return (
            (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x0) / \
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)
        )

    def _scene_ids_tensor(self, scene_ids, num_boxes, device):
        if scene_ids is None:
            return torch.zeros(num_boxes, dtype=torch.long, device=device)
        if isinstance(scene_ids, torch.Tensor):
            return scene_ids.to(device=device, dtype=torch.long)
        return torch.as_tensor(scene_ids, dtype=torch.long, device=device)

    def _denormalize_box_params(self, box_params):
        if self.box_stats is None:
            raise ValueError('train_stats_file must be set for inference-time physical guidance.')
        stats = torch.as_tensor(self.box_stats, dtype=box_params.dtype, device=box_params.device)
        min_lhw, max_lhw = stats[:3], stats[3:6]
        min_xyz, max_xyz = stats[6:9], stats[9:12]

        sizes = ((box_params[:, :3] + 1.0) / 2.0) * (max_lhw - min_lhw) + min_lhw
        translations = ((box_params[:, 3:6] + 1.0) / 2.0) * (max_xyz - min_xyz) + min_xyz
        angles = postprocess_sincos2arctan(box_params[:, -2:])
        return torch.cat((sizes, translations, angles), dim=-1)

    def _boxes_to_axis_aligned_corners(self, denorm_boxes):
        half_sizes = denorm_boxes[:, :3].clamp(min=1e-4) * 0.5
        centers = denorm_boxes[:, 3:6]
        return torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)

    def _guidance_enabled(self):
        return bool(cfg_get(self.inference_guidance, 'enabled', False))

    def _is_constraint_active(self, constraint_cfg, denoise_progress, denoise_step):
        if constraint_cfg is None:
            return False
        if not bool(cfg_get(constraint_cfg, 'enabled', True)):
            return False
        global_interval = max(int(cfg_get(self.inference_guidance, 'interval', 1)), 1)
        global_start_ratio = min(max(float(cfg_get(self.inference_guidance, 'start_ratio', 0.0)), 0.0), 1.0)
        
        c_start_ratio = cfg_get(constraint_cfg, 'start_ratio', None)
        if c_start_ratio is not None:
            c_start_ratio = min(max(float(c_start_ratio), 0.0), 1.0)
        else:
            c_start_ratio = global_start_ratio

        c_interval = cfg_get(constraint_cfg, 'interval', None)
        if c_interval is not None:
            c_interval = max(int(c_interval), 1)
        else:
            c_interval = global_interval

        return denoise_progress >= c_start_ratio and denoise_step % c_interval == 0

    def _guidance_active_for_timestep(self, timestep):
        if not self._guidance_enabled():
            return False
        denoise_step = (self.num_timesteps - 1) - int(timestep)
        if self.num_timesteps > 1:
            denoise_progress = denoise_step / float(self.num_timesteps - 1)
        else:
            denoise_progress = 1.0

        constraints_cfg = cfg_get(self.inference_guidance, 'constraints', None)
        if constraints_cfg is not None:
            # Check collision
            collision_cfg = self._collision_guidance_cfg()
            if collision_cfg and bool(cfg_get(collision_cfg, 'enabled', False)):
                if self._is_constraint_active(collision_cfg, denoise_progress, denoise_step):
                    return True

            # Check room_outer
            room_outer_cfg = cfg_get(constraints_cfg, 'room_outer', None)
            if room_outer_cfg and float(cfg_get(room_outer_cfg, 'weight', 0.0)) > 0:
                if self._is_constraint_active(room_outer_cfg, denoise_progress, denoise_step):
                    return True

            # Check walkable
            walkable_cfg = cfg_get(constraints_cfg, 'walkable', None)
            if walkable_cfg and bool(cfg_get(walkable_cfg, 'enabled', False)):
                if self._is_constraint_active(walkable_cfg, denoise_progress, denoise_step):
                    return True

            # Check relational
            relational_cfg = cfg_get(constraints_cfg, 'relational', None)
            if relational_cfg and bool(cfg_get(relational_cfg, 'enabled', False)):
                if self._is_constraint_active(relational_cfg, denoise_progress, denoise_step):
                    return True

            return False

        global_interval = max(int(cfg_get(self.inference_guidance, 'interval', 1)), 1)
        global_start_ratio = min(max(float(cfg_get(self.inference_guidance, 'start_ratio', 0.0)), 0.0), 1.0)
        return denoise_progress >= global_start_ratio and denoise_step % global_interval == 0

    def _collision_guidance_cfg(self):
        constraints_cfg = cfg_get(self.inference_guidance, 'constraints', None)
        return cfg_get(constraints_cfg, 'collision', None)

    def _compute_pairwise_penetration(self, scene_boxes):
        sizes = scene_boxes[:, :3].clamp(min=1e-4)
        dx, dy, dz = sizes[:, 0], sizes[:, 1], sizes[:, 2]
        angle = scene_boxes[:, 6]
        
        cos_a = torch.abs(torch.cos(angle))
        sin_a = torch.abs(torch.sin(angle))
        
        new_dx = dx * cos_a + dz * sin_a
        new_dz = dx * sin_a + dz * cos_a
        
        half_sizes = torch.stack([new_dx, dy, new_dz], dim=-1) * 0.5
        
        centers = scene_boxes[:, 3:6]
        center_delta = torch.abs(centers[:, None, :] - centers[None, :, :])
        overlap_delta = torch.relu(half_sizes[:, None, :] + half_sizes[None, :, :] - center_delta)
        penetration_volume = overlap_delta[..., 0] * overlap_delta[..., 1] * overlap_delta[..., 2]
        max_penetration_depth = overlap_delta.max(dim=-1).values
        return penetration_volume, max_penetration_depth

    def _compute_collision_guidance_loss(self, box_params, scene_ids, ignore_enabled=False, objectness=None):
        collision_cfg = self._collision_guidance_cfg()
        if not ignore_enabled and (collision_cfg is None or not bool(cfg_get(collision_cfg, 'enabled', False))):
            return None, {
                'constraint': 'collision',
                'skipped': True,
                'skip_reason': 'constraint_disabled',
            }
        if self.box_stats is None:
            return None, {
                'constraint': 'collision',
                'skipped': True,
                'skip_reason': 'missing_train_stats',
            }

        metric = str(cfg_get(collision_cfg, 'metric', 'aabb') if collision_cfg is not None else 'aabb').lower()

        iou_threshold = float(cfg_get(collision_cfg, 'iou_threshold', 0.0) if collision_cfg is not None else 0.0)
        penetration_threshold = float(cfg_get(collision_cfg, 'penetration_threshold', 0.0) if collision_cfg is not None else 0.0)
        iou_weight = float(cfg_get(collision_cfg, 'iou_weight', 1.0) if collision_cfg is not None else 1.0)
        penetration_weight = float(cfg_get(collision_cfg, 'penetration_weight', 0.0) if collision_cfg is not None else 0.0)
        denorm_boxes = self._denormalize_box_params(box_params)
        scene_ids = self._scene_ids_tensor(scene_ids, box_params.shape[0], box_params.device)
        # Modify to remove scene/floor
        if objectness is None:
            objectness = torch.ones(box_params.shape[0], dtype=torch.bool, device=box_params.device)
        else:
            objectness = objectness.to(device=box_params.device, dtype=torch.bool)
        scene_losses = []
        per_scene_mean_ious = []
        per_scene_max_ious = []
        per_scene_penetration = []
        per_scene_max_penetration = []
        total_pairs = 0
        total_pairs_above_threshold = 0
        total_pairs_with_penetration = 0

        for scene_id in torch.unique(scene_ids):
            scene_mask = (scene_ids == scene_id) & objectness 
            if int(scene_mask.sum().item()) < 2:
                continue

            scene_boxes = denorm_boxes[scene_mask]
            
            # Clamp sizes to prevent negative volumes which zero out the IoU
            clamped_sizes = scene_boxes[:, :3].clamp(min=1e-4)
            clamped_scene_boxes = torch.cat([clamped_sizes, scene_boxes[:, 3:]], dim=-1)

            # Map to cal_iou_3d format: (x, y, z, w, h, l, alpha) -> (3, 5, 4, 0, 2, 1, 6)
            mapped_boxes = clamped_scene_boxes[:, [3, 5, 4, 0, 2, 1, 6]]
            mapped_boxes1 = mapped_boxes.unsqueeze(1).repeat(1, mapped_boxes.size(0), 1)
            mapped_boxes2 = mapped_boxes.unsqueeze(0).repeat(mapped_boxes.size(0), 1, 1)
            
            iou_matrix = cal_iou_3d(mapped_boxes1, mapped_boxes2)
            iou_matrix = torch.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            pair_mask = torch.triu(
                torch.ones_like(iou_matrix, dtype=torch.bool),
                diagonal=1,
            )
            
            # Filter out collisions involving floor/_scene_ if objectness mask is provided
            if getattr(self, 'objectness', None) is not None:
                scene_objectness = self.objectness[scene_mask]
                valid_pairs = scene_objectness.unsqueeze(1) & scene_objectness.unsqueeze(0)
                pair_mask = pair_mask & valid_pairs

            pair_iou = iou_matrix[pair_mask]
            penetration_matrix, penetration_depth_matrix = self._compute_pairwise_penetration(scene_boxes)
            pair_penetration = penetration_matrix[pair_mask]
            pair_penetration_depth = penetration_depth_matrix[pair_mask]
            if pair_iou.numel() == 0:
                continue

            active_pair_iou = torch.relu(pair_iou - iou_threshold)
            active_pair_penetration = torch.relu(pair_penetration - penetration_threshold)
            total_pairs += int(pair_iou.numel())
            total_pairs_above_threshold += int((pair_iou > iou_threshold).sum().item())
            total_pairs_with_penetration += int((pair_penetration > penetration_threshold).sum().item())
            per_scene_mean_ious.append(pair_iou.mean())
            per_scene_max_ious.append(pair_iou.max())
            per_scene_penetration.append(pair_penetration.mean())
            per_scene_max_penetration.append(pair_penetration_depth.max())

            scene_loss_terms = []
            if iou_weight > 0.0 and torch.any(active_pair_iou > 0):
                scene_loss_terms.append(iou_weight * active_pair_iou.mean())
            if penetration_weight > 0.0 and torch.any(active_pair_penetration > 0):
                scene_loss_terms.append(penetration_weight * active_pair_penetration.mean())
            if len(scene_loss_terms) > 0:
                scene_losses.append(torch.stack(scene_loss_terms).sum())

        if len(per_scene_mean_ious) == 0:
            return None, {
                'constraint': 'collision',
                'skipped': True,
                'skip_reason': 'insufficient_pairs',
                'num_pairs': 0,
                'pairs_above_threshold': 0,
                'pairs_with_penetration': 0,
                'avg_scene_iou': 0.0,
                'max_pair_iou': 0.0,
                'avg_scene_penetration': 0.0,
                'max_pair_penetration': 0.0,
                'iou_threshold': iou_threshold,
                'penetration_threshold': penetration_threshold,
                'metric': metric,
            }

        if len(scene_losses) == 0:
            return None, {
                'constraint': 'collision',
                'skipped': True,
                'skip_reason': 'below_threshold',
                'num_pairs': total_pairs,
                'pairs_above_threshold': total_pairs_above_threshold,
                'pairs_with_penetration': total_pairs_with_penetration,
                'avg_scene_iou': float(torch.stack(per_scene_mean_ious).mean().detach().item()),
                'max_pair_iou': float(torch.stack(per_scene_max_ious).max().detach().item()),
                'avg_scene_penetration': float(torch.stack(per_scene_penetration).mean().detach().item()),
                'max_pair_penetration': float(torch.stack(per_scene_max_penetration).max().detach().item()),
                'iou_threshold': iou_threshold,
                'penetration_threshold': penetration_threshold,
                'iou_weight': iou_weight,
                'penetration_weight': penetration_weight,
                'metric': metric,
            }

        loss = torch.stack(scene_losses).mean()
        stats = {
            'constraint': 'collision',
            'skipped': False,
            'num_pairs': total_pairs,
            'pairs_above_threshold': total_pairs_above_threshold,
            'pairs_with_penetration': total_pairs_with_penetration,
            'avg_scene_iou': float(torch.stack(per_scene_mean_ious).mean().detach().item()),
            'max_pair_iou': float(torch.stack(per_scene_max_ious).max().detach().item()),
            'avg_scene_penetration': float(torch.stack(per_scene_penetration).mean().detach().item()),
            'max_pair_penetration': float(torch.stack(per_scene_max_penetration).max().detach().item()),
            'iou_threshold': iou_threshold,
            'penetration_threshold': penetration_threshold,
            'iou_weight': iou_weight,
            'penetration_weight': penetration_weight,
            'metric': metric,
        }
        return loss, stats

    def _apply_inference_guidance(self, pred_xstart, model_mean, model_variance, timestep, scene_ids, triples=None, floor_plan=None, room_outer_box=None, objectness=None):
        step_stats = {
            'timestep': int(timestep),
            'applied': False,
        }
        if not self._guidance_active_for_timestep(timestep):
            step_stats['skip_reason'] = 'schedule_inactive'
            return model_mean, step_stats

        denoise_step = (self.num_timesteps - 1) - int(timestep)
        if self.num_timesteps > 1:
            denoise_progress = denoise_step / float(self.num_timesteps - 1)
        else:
            denoise_progress = 1.0

        collision_cfg = self._collision_guidance_cfg()
        
        # Get individual weights from config (falling back to original defaults)
        constraints_cfg = cfg_get(self.inference_guidance, 'constraints', {}) or {}
        room_outer_cfg = cfg_get(constraints_cfg, 'room_outer', {}) or {}
        walkable_cfg = cfg_get(constraints_cfg, 'walkable', {}) or {}
        relational_cfg = cfg_get(constraints_cfg, 'relational', {}) or {}
        
        collision_weight = float(cfg_get(collision_cfg, 'weight', 10.0)) if collision_cfg is not None else 10.0
        room_outer_weight = float(cfg_get(room_outer_cfg, 'weight', 10.0))
        walkable_weight = float(cfg_get(walkable_cfg, 'weight', 1.0))
        
        # Keep strength as a global multiplier for backward compatibility
        strength = float(cfg_get(collision_cfg, 'strength', 0.0)) if collision_cfg is not None else 0.0
        if strength <= 0.0:
            strength = float(cfg_get(self.inference_guidance, 'strength', 20.0))
        if strength <= 0.0:
            step_stats['skip_reason'] = 'zero_strength'
            return model_mean, step_stats

        # Check active status for each individual constraint for this timestep
        collision_active = self._is_constraint_active(collision_cfg, denoise_progress, denoise_step) if collision_cfg and bool(cfg_get(collision_cfg, 'enabled', False)) else False
        room_outer_active = self._is_constraint_active(room_outer_cfg, denoise_progress, denoise_step) if room_outer_cfg and room_outer_weight > 0 else False
        walkable_active = self._is_constraint_active(walkable_cfg, denoise_progress, denoise_step) if walkable_cfg and bool(cfg_get(walkable_cfg, 'enabled', False)) and walkable_weight > 0 else False
        relational_active = self._is_constraint_active(relational_cfg, denoise_progress, denoise_step) if relational_cfg and bool(cfg_get(relational_cfg, 'enabled', False)) else False

        # Collision loss computation
        if collision_active:
            collision_loss, collision_stats = self._compute_collision_guidance_loss(pred_xstart, scene_ids, objectness=objectness)
            step_stats.update(collision_stats)
        else:
            collision_loss = None
            step_stats.update({
                'constraint': 'collision',
                'skipped': True,
                'skip_reason': 'schedule_inactive',
            })
        
        need_denorm = room_outer_active or relational_active or walkable_active
        if need_denorm:
            denorm_boxes = self._denormalize_box_params(pred_xstart)
        else:
            denorm_boxes = None

        # Room outer loss computation
        room_outer_loss = 0.0
        if room_outer_active and denorm_boxes is not None:
            room_outer_loss = compute_room_outer_loss(denorm_boxes, room_outer_box, scene_ids, objectness)
        
        # Directional & Support Relational Guidance Loss
        relational_loss = 0.0
        rel_raw_val = 0.0
        if relational_active and denorm_boxes is not None:
            rel_w = float(cfg_get(relational_cfg, 'weight', 10.0))
            margin = float(cfg_get(relational_cfg, 'margin', 0.05))
            close_th = float(cfg_get(relational_cfg, 'close_threshold', 0.45))
            stand_th = float(cfg_get(relational_cfg, 'stand_threshold', 0.04))
            rel_val = compute_relational_guidance_loss(
                denorm_boxes, triples, 
                predicate_names=None, objectness=objectness, 
                margin=margin, close_threshold=close_th, stand_threshold=stand_th
            )
            rel_raw_val = rel_val.detach() if isinstance(rel_val, torch.Tensor) else float(rel_val)
            relational_loss = rel_val * rel_w

        components_cfg = cfg_get(walkable_cfg, 'components', None) if walkable_cfg else None
        
        walkable_loss = 0.0
        c_center_loss = 0.0
        c_path_loss = 0.0
        c1_loss = 0.0
        c2_loss = 0.0

        if walkable_active and denorm_boxes is not None:
            if components_cfg is not None:
                # --- MODULAR MULTI-COMPONENT WALKABLE LOSS SYSTEM ---
                # 1. Center Penalty Sub-Component
                cp_cfg = cfg_get(components_cfg, 'center_penalty', None)
                if cp_cfg and bool(cfg_get(cp_cfg, 'enabled', False)):
                    cp_w = float(cfg_get(cp_cfg, 'weight', 1.0))
                    sigma = float(cfg_get(cp_cfg, 'sigma', 0.5))
                    cp_val = compute_center_penalty_loss(denorm_boxes, objectness=objectness, sigma=sigma)
                    c_center_loss = cp_val.detach() if isinstance(cp_val, torch.Tensor) else float(cp_val)
                    walkable_loss = walkable_loss + cp_val * cp_w

                # 2. Pathfinding Sub-Component
                pf_cfg = cfg_get(components_cfg, 'pathfinding', None)
                if pf_cfg and bool(cfg_get(pf_cfg, 'enabled', False)):
                    pf_w = float(cfg_get(pf_cfg, 'weight', 1.0))
                    rw = float(cfg_get(pf_cfg, 'robot_width_real', 0.5))
                    rh = float(cfg_get(pf_cfg, 'robot_hight_real', 1.5))
                    pf_val = compute_pathfinding_walkable_loss(
                        denorm_boxes, floor_plan, objectness=objectness,
                        robot_width_real=rw, robot_hight_real=rh
                    )
                    c_path_loss = pf_val.detach() if isinstance(pf_val, torch.Tensor) else float(pf_val)
                    walkable_loss = walkable_loss + pf_val * pf_w

                # 3. Edge-Gaussian Sub-Component
                eg_cfg = cfg_get(components_cfg, 'edge_gaussian', None)
                if eg_cfg and bool(cfg_get(eg_cfg, 'enabled', False)):
                    eg_w = float(cfg_get(eg_cfg, 'weight', 1.0))
                    rw = float(cfg_get(eg_cfg, 'robot_width_real', 0.35))
                    rh = float(cfg_get(eg_cfg, 'robot_hight_real', 1.5))
                    sigma_scale = float(cfg_get(eg_cfg, 'sigma_scale', 0.5))
                    hm_w = float(cfg_get(eg_cfg, 'heatmap_weight', 0.8))
                    rep_w = float(cfg_get(eg_cfg, 'repulsion_weight', 0.2))
                    eg_val, comp_dict = compute_edge_gaussian_walkable_loss(
                        denorm_boxes, floor_plan, objectness=objectness,
                        robot_width_real=rw, robot_hight_real=rh,
                        sigma_scale=sigma_scale, heatmap_weight=hm_w, repulsion_weight=rep_w,
                        return_components=True, verbose=False
                    )
                    c1_loss = comp_dict['c1_floor_heatmap'].detach() if isinstance(comp_dict['c1_floor_heatmap'], torch.Tensor) else float(comp_dict['c1_floor_heatmap'])
                    c2_loss = comp_dict['c2_pairwise_repulsion'].detach() if isinstance(comp_dict['c2_pairwise_repulsion'], torch.Tensor) else float(comp_dict['c2_pairwise_repulsion'])
                    walkable_loss = walkable_loss + eg_val * eg_w
            else:
                # --- LEGACY SINGLE TYPE FALLBACK ---
                robot_width_real = float(cfg_get(walkable_cfg, 'robot_width_real', 0.35))
                robot_hight_real = float(cfg_get(walkable_cfg, 'robot_hight_real', 1.5))
                walkable_type = str(cfg_get(walkable_cfg, 'type', 'pathfinding')).lower() if walkable_cfg else 'pathfinding'
                if walkable_type == 'edge_gaussian':
                    sigma_scale = float(cfg_get(walkable_cfg, 'sigma_scale', 0.5))
                    heatmap_weight = float(cfg_get(walkable_cfg, 'heatmap_weight', 0.8))
                    repulsion_weight = float(cfg_get(walkable_cfg, 'repulsion_weight', 0.2))
                    walkable_loss, comp_dict = compute_edge_gaussian_walkable_loss(
                        denorm_boxes, floor_plan, objectness=objectness,
                        robot_width_real=robot_width_real, robot_hight_real=robot_hight_real,
                        sigma_scale=sigma_scale, heatmap_weight=heatmap_weight, repulsion_weight=repulsion_weight,
                        return_components=True, verbose=False
                    )
                    c1_loss = comp_dict['c1_floor_heatmap'].detach() if isinstance(comp_dict['c1_floor_heatmap'], torch.Tensor) else float(comp_dict['c1_floor_heatmap'])
                    c2_loss = comp_dict['c2_pairwise_repulsion'].detach() if isinstance(comp_dict['c2_pairwise_repulsion'], torch.Tensor) else float(comp_dict['c2_pairwise_repulsion'])
                elif walkable_type == 'center_penalty':
                    cp_val = compute_center_penalty_loss(denorm_boxes, objectness=objectness)
                    c_center_loss = cp_val.detach() if isinstance(cp_val, torch.Tensor) else float(cp_val)
                    walkable_loss = cp_val
                else:
                    pf_val = compute_pathfinding_walkable_loss(
                        denorm_boxes, floor_plan, objectness=objectness,
                        robot_width_real=robot_width_real, robot_hight_real=robot_hight_real
                    )
                    c_path_loss = pf_val.detach() if isinstance(pf_val, torch.Tensor) else float(pf_val)
                    walkable_loss = pf_val
        
        # Combine total guidance loss
        total_guidance_loss = 0.0
        if collision_loss is not None:
            total_guidance_loss = total_guidance_loss + collision_loss * collision_weight
            
        if room_outer_loss is not None and (not isinstance(room_outer_loss, float) or room_outer_loss > 0):
            total_guidance_loss = total_guidance_loss + room_outer_loss * room_outer_weight
            
        if walkable_loss is not None and (not isinstance(walkable_loss, float) or walkable_loss > 0):
            total_guidance_loss = total_guidance_loss + walkable_loss * walkable_weight
            
        if relational_loss is not None and (not isinstance(relational_loss, float) or relational_loss > 0):
            total_guidance_loss = total_guidance_loss + relational_loss

        # Ensure that if all losses were effectively 0 or None, we don't try to compute grads if not required
        if isinstance(total_guidance_loss, float) and total_guidance_loss == 0.0:
            step_stats['skip_reason'] = step_stats.get('skip_reason', 'zero_loss')
            return model_mean, step_stats
        elif total_guidance_loss is None or not torch.isfinite(total_guidance_loss):
            step_stats['skip_reason'] = step_stats.get('skip_reason', 'invalid_loss')
            return model_mean, step_stats

        # Compute gradient using total_guidance_loss instead of just collision_loss
        guidance_grad_tuple = torch.autograd.grad(total_guidance_loss, pred_xstart, allow_unused=True)
        guidance_grad = guidance_grad_tuple[0]
        
        if guidance_grad is None:
            step_stats['skip_reason'] = step_stats.get('skip_reason', 'unused_grad')
            return model_mean, step_stats
            
        grad_norm = guidance_grad.norm(p=2, dim=1, keepdim=True)
        grad_clip = float(cfg_get(self.inference_guidance, 'grad_clip', 0.0))
        if grad_clip > 0.0:
            clip_scale = torch.clamp(grad_clip / (grad_norm + 1e-12), max=1.0)
            guidance_grad = guidance_grad * clip_scale

        variance_preconditioned_grad = model_variance * guidance_grad
        guided_mean = model_mean - strength * variance_preconditioned_grad

        step_stats.update({
            'applied': True,
            'guidance_strength': strength,
            'collision_loss': collision_loss.detach() if collision_loss is not None else 0.0,
            'room_outer_loss': room_outer_loss.detach() if isinstance(room_outer_loss, torch.Tensor) else float(room_outer_loss),
            'walkable_loss': walkable_loss.detach() if isinstance(walkable_loss, torch.Tensor) else float(walkable_loss),
            'relational_loss': rel_raw_val,
            'walkable_center_penalty': c_center_loss,
            'walkable_pathfinding': c_path_loss,
            'walkable_c1_heatmap': c1_loss,
            'walkable_c2_repulsion': c2_loss,
            'variance_scale_mean': model_variance.mean().detach(),
            'variance_scale_max': model_variance.max().detach(),
            'grad_norm_mean': grad_norm.mean().detach(),
            'grad_norm_max': grad_norm.max().detach(),
        })
        return guided_mean.detach(), step_stats

    def _summarize_guidance_stats(self, step_stats, final_sample, scene_ids, objectness=None):
        summary = {
            'enabled': self._guidance_enabled(),
            'scheduled_steps': len(step_stats),
            'applied_steps': sum(1 for stat in step_stats if stat.get('applied')),
            'step_stats': step_stats,
        }

        collision_cfg = self._collision_guidance_cfg()
        summary['guidance_strength'] = float(cfg_get(collision_cfg, 'strength', 0.0)) if collision_cfg is not None else 0.0
        summary['interval'] = int(cfg_get(self.inference_guidance, 'interval', 1)) if self._guidance_enabled() else 0
        summary['start_ratio'] = float(cfg_get(self.inference_guidance, 'start_ratio', 0.0)) if self._guidance_enabled() else 0.0
        
        relational_cfg = cfg_get(self.inference_guidance, 'constraints', {}).get('relational', {}) if self._guidance_enabled() and self.inference_guidance.get('constraints') else {}
        rel_start_ratio = cfg_get(relational_cfg, 'start_ratio', None)
        if rel_start_ratio is not None:
            summary['relational_start_ratio'] = float(rel_start_ratio)

        def _to_float(v):
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v) if v is not None else 0.0

        applied_stats = [stat for stat in step_stats if stat.get('applied')]
        if applied_stats:
            summary['avg_guided_scene_iou'] = float(np.mean([_to_float(stat.get('avg_scene_iou', 0.0)) for stat in applied_stats]))
            summary['avg_guided_scene_penetration'] = float(np.mean([_to_float(stat.get('avg_scene_penetration', 0.0)) for stat in applied_stats]))
            summary['avg_guided_collision_loss'] = float(np.mean([_to_float(stat.get('collision_loss', 0.0)) for stat in applied_stats]))
            summary['avg_guided_room_outer_loss'] = float(np.mean([_to_float(stat.get('room_outer_loss', 0.0)) for stat in applied_stats]))
            summary['avg_guided_walkable_loss'] = float(np.mean([_to_float(stat.get('walkable_loss', 0.0)) for stat in applied_stats]))
            summary['avg_guided_center_penalty'] = float(np.mean([_to_float(stat.get('walkable_center_penalty', 0.0)) for stat in applied_stats]))
            summary['avg_guided_c1_heatmap'] = float(np.mean([_to_float(stat.get('walkable_c1_heatmap', 0.0)) for stat in applied_stats]))
            summary['avg_guided_c2_repulsion'] = float(np.mean([_to_float(stat.get('walkable_c2_repulsion', 0.0)) for stat in applied_stats]))
            summary['avg_guided_relational_loss'] = float(np.mean([_to_float(stat.get('relational_loss', 0.0)) for stat in applied_stats]))
            summary['avg_guided_grad_norm'] = float(np.mean([_to_float(stat.get('grad_norm_mean', 0.0)) for stat in applied_stats]))
            summary['avg_guided_variance_scale'] = float(np.mean([_to_float(stat.get('variance_scale_mean', 0.0)) for stat in applied_stats]))
        else:
            summary['avg_guided_scene_iou'] = 0.0
            summary['avg_guided_scene_penetration'] = 0.0
            summary['avg_guided_collision_loss'] = 0.0
            summary['avg_guided_room_outer_loss'] = 0.0
            summary['avg_guided_walkable_loss'] = 0.0
            summary['avg_guided_center_penalty'] = 0.0
            summary['avg_guided_c1_heatmap'] = 0.0
            summary['avg_guided_c2_repulsion'] = 0.0
            summary['avg_guided_relational_loss'] = 0.0
            summary['avg_guided_grad_norm'] = 0.0
            summary['avg_guided_variance_scale'] = 0.0

        try:
            _, final_collision_stats = self._compute_collision_guidance_loss(final_sample.detach(), scene_ids, ignore_enabled=True, objectness=objectness)
            summary['final_avg_scene_iou'] = float(final_collision_stats.get('avg_scene_iou', 0.0))
            summary['final_max_pair_iou'] = float(final_collision_stats.get('max_pair_iou', 0.0))
            summary['final_avg_scene_penetration'] = float(final_collision_stats.get('avg_scene_penetration', 0.0))
            summary['final_max_pair_penetration'] = float(final_collision_stats.get('max_pair_penetration', 0.0))
            summary['final_pairs_above_threshold'] = int(final_collision_stats.get('pairs_above_threshold', 0))
            summary['final_pairs_with_penetration'] = int(final_collision_stats.get('pairs_with_penetration', 0))
            summary['final_num_pairs'] = int(final_collision_stats.get('num_pairs', 0))
        except ValueError as exc:
            summary['final_avg_scene_iou'] = 0.0
            summary['final_max_pair_iou'] = 0.0
            summary['final_avg_scene_penetration'] = 0.0
            summary['final_max_pair_penetration'] = 0.0
            summary['final_pairs_above_threshold'] = 0
            summary['final_pairs_with_penetration'] = 0
            summary['final_num_pairs'] = 0
            summary['final_metrics_error'] = str(exc)

        return summary

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, condition, condition_cross, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_sg(self, denoise_fn, data, t, obj_embed, triples, condition, scene_ids=None, noise_fn=torch.randn, clip_denoised=False, return_pred_xstart=False, floor_plan=None, room_outer_box=None, objectness=None):
        """
        Sample from the model
        """
        guidance_step_stats = {
            'timestep': int(t[0].item()),
            'applied': False,
            'skip_reason': 'guidance_disabled',
        }
        if self._guidance_enabled():
            with torch.enable_grad():
                guided_data = data.detach().clone().requires_grad_(True)
                model_mean, model_variance, model_log_variance, pred_xstart = self.p_mean_variance(
                    denoise_fn,
                    data=guided_data,
                    t=t,
                    obj_embed=obj_embed,
                    triples=triples,
                    condition=condition,
                    clip_denoised=clip_denoised,
                    return_pred_xstart=True,
                )
                # [MODIFIED] Pass floor_plan and room_outer_box down to apply_inference_guidance
                model_mean, guidance_step_stats = self._apply_inference_guidance(
                    pred_xstart=pred_xstart,
                    model_mean=model_mean,
                    model_variance=model_variance,
                    timestep=int(t[0].item()),
                    scene_ids=scene_ids,
                    triples=triples,
                    floor_plan=floor_plan,
                    room_outer_box=room_outer_box,
                    objectness=objectness
                )
                pred_xstart = pred_xstart.detach()
                model_log_variance = model_log_variance.detach()
        else:
            model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
                denoise_fn,
                data=data,
                t=t,
                obj_embed=obj_embed,
                triples=triples,
                condition=condition,
                clip_denoised=clip_denoised,
                return_pred_xstart=True,
            )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        if return_pred_xstart:
            return sample, pred_xstart, guidance_step_stats
        return sample, guidance_step_stats


    def p_sample_loop(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_sg(self, denoise_fn, shape, device, obj_embed, triples, condition, scene_ids=None, noise_fn=torch.randn, clip_denoised=True, keep_running=False, floor_plan=None, room_outer_box=None,objectness=None):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        x_t = noise_fn(size=shape, dtype=torch.float, device=device)
        step_stats = []
        for t in tqdm(reversed(range(0, self.num_timesteps if not keep_running else len(self.betas)))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            # [MODIFIED] Pass floor_plan and room_outer_box
            x_t, guidance_step_stats = self.p_sample_sg(denoise_fn=denoise_fn, data=x_t, t=t_, obj_embed=obj_embed, triples=triples, condition=condition, scene_ids=scene_ids, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False, floor_plan=floor_plan, room_outer_box=room_outer_box, objectness=objectness)
            
            # [MODIFIED] Freeze non-object nodes (e.g. floor, _scene_) using their Ground Truth values
            if getattr(self, 'objectness', None) is not None and getattr(self, 'gt_boxes', None) is not None:
                mask = ~self.objectness.bool()
                if mask.any():
                    if t > 0:
                        noise = torch.randn_like(self.gt_boxes)
                        noised_gt = self.q_sample(x_start=self.gt_boxes, t=t_, noise=noise)
                    else:
                        noised_gt = self.gt_boxes
                    x_t[mask] = noised_gt[mask]

            if self._guidance_enabled():
                step_stats.append(guidance_step_stats)

        assert x_t.shape == shape
        self.latest_sampling_stats = self._summarize_guidance_stats(step_stats, x_t, scene_ids, objectness=objectness)
        if self._guidance_enabled() and self.latest_sampling_stats.get('applied_steps', 0) > 0:
            s = self.latest_sampling_stats
            print(f"[Inference Guidance Summary] Applied Steps: {s['applied_steps']}/{s['scheduled_steps']} | Collision Loss: {s['avg_guided_collision_loss']:.4f} | Room Outer Loss: {s['avg_guided_room_outer_loss']:.4f} | Walkable Loss: {s['avg_guided_walkable_loss']:.4f} (Center Penalty: {s['avg_guided_center_penalty']:.4f}, C1 Heatmap: {s['avg_guided_c1_heatmap']:.4f}, C2 Repulsion: {s['avg_guided_c2_repulsion']:.4f}) | Relational Loss: {s['avg_guided_relational_loss']:.4f}")
        return x_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, condition, condition_cross,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)

            # [MODIFIED] Freeze non-object nodes (e.g. floor, _scene_) using their Ground Truth values
            if getattr(self, 'objectness', None) is not None and getattr(self, 'gt_boxes', None) is not None:
                mask = ~self.objectness.bool()
                if mask.any():
                    if t > 0:
                        noise = torch.randn_like(self.gt_boxes)
                        noised_gt = self.q_sample(x_start=self.gt_boxes, t=t_, noise=noise)
                    else:
                        noised_gt = self.gt_boxes
                    img_t[mask] = noised_gt[mask]

            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs


    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, condition, condition_cross, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def IoU_loss(self, data_t,  timestep=None, pred_data=None, scene_ids=None):
        # get x_recon & valid mask
        if self.model_mean_type == 'eps':
            self.x_recon = self._predict_xstart_from_eps(data_t, timestep, eps=pred_data)
        else:
            self.x_recon = pred_data

        # descale bounding box to world coordinate system
        descale_bbox = descale_box_params(self.x_recon, file=self.bbox_norm_file, angle=False)
        angle = postprocess_sincos2arctan(self.x_recon[:,-2:])
        descale_bbox = torch.concat((descale_bbox[:,:-2],angle),dim=-1)
        # Map to cal_iou_3d format: (x, y, z, w, h, l, alpha) -> (3, 5, 4, 0, 2, 1, 6)
        mapped_bbox = descale_bbox[:, [3, 5, 4, 0, 2, 1, 6]]
        mapped_boxes1 = mapped_bbox.unsqueeze(1).repeat(1, mapped_bbox.size(0), 1)
        mapped_boxes2 = mapped_bbox.unsqueeze(0).repeat(mapped_bbox.size(0), 1, 1)
        bbox_iou = cal_iou_3d(mapped_boxes1, mapped_boxes2)

        bbox_iou = torch.where(torch.isnan(bbox_iou), torch.zeros_like(bbox_iou), bbox_iou)

        # get the iou loss weight w.r.t time
        w_iou = self._extract(self.alphas_cumprod.to(data_t.device), timestep, bbox_iou.shape)
        # only consider bboxes in the same scenes
        assert scene_ids is not None
        scene_ids = torch.tensor(scene_ids, dtype=torch.int64, device=data_t.device)
        scene_mask = scene_ids[:, None] == scene_ids
        diag_mask = torch.eye(scene_mask.size(0), dtype=torch.bool, device=scene_mask.device)
        scene_mask[diag_mask] = False  # remove the diagomal values

        # [MODIFIED] Do not compute IoU loss for floor/_scene_
        if getattr(self, 'objectness', None) is not None:
            obj_mask = self.objectness.to(device=scene_mask.device, dtype=torch.bool)
            valid_pairs = obj_mask.unsqueeze(1) & obj_mask.unsqueeze(0)
            scene_mask = scene_mask & valid_pairs

        iou_indices = torch.where(scene_mask)
        w_iou_selected = w_iou[iou_indices[0]].reshape(-1)
        if not torch.isnan(bbox_iou[iou_indices]).any():
            bbox_iou_valid = bbox_iou[iou_indices] + 1e-6
        else:
            bbox_iou_valid = torch.zeros(len(w_iou_selected)).to(data_t.device) # meaningful bbox_iou in the same scene.
            print("bbox_iou is NaN")
        loss_iou_valid = w_iou_selected * 0.5 * bbox_iou_valid
        return loss_iou_valid, bbox_iou_valid

    def SDFusion_loss(self, data_t, t, denoise_out, target, scene_ids):
        loss_size = torch.nn.functional.mse_loss(target[:, 0:self.size_dim], denoise_out[:, 0:self.size_dim], reduction='none').mean(
            dim=list(range(1, len(data_t.shape))))
        loss_trans = torch.nn.functional.mse_loss(target[:, self.size_dim:self.size_dim + self.translation_dim], denoise_out[:, self.size_dim:self.size_dim + self.translation_dim], reduction='none').mean(dim=list(range(1, len(data_t.shape))))
        loss_angle = torch.nn.functional.mse_loss(target[:, self.size_dim + self.translation_dim:self.bbox_dim], denoise_out[:, self.size_dim + self.translation_dim:self.bbox_dim], reduction='none').mean(
            dim=list(range(1, len(data_t.shape))))
        loss_bbox = torch.nn.functional.mse_loss(target, denoise_out, reduction='none').mean(dim=list(range(1, len(data_t.shape))))
        logvar_t = self.logvar[t].to(data_t.device)
        loss = loss_bbox / torch.exp(logvar_t) + logvar_t
        if self.loss_iou:
            loss_iou_valid, bbox_iou_valid = self.IoU_loss(data_t,  timestep=t, pred_data=denoise_out, scene_ids=scene_ids)
        else:
            loss_iou_valid = torch.zeros(len(denoise_out)).to(data_t.device)
            bbox_iou_valid = torch.zeros(len(denoise_out)).to(data_t.device)
        return loss.mean() + loss_iou_valid.mean(), {
            'loss.bbox': loss_bbox.mean(),
            'loss.trans': loss_trans.mean(),
            'loss.size': loss_size.mean(),
            'loss.angle': loss_angle.mean(),
            'loss.liou': loss_iou_valid.mean(),
            'loss.bbox_iou': bbox_iou_valid.mean(),
        }

    def diffusion_loss(self, data_t, t, denoise_out, target, scene_ids):
        loss_size = ((target[:, 0:self.size_dim] - denoise_out[:, 0:self.size_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_trans = ((target[:, self.size_dim:self.size_dim + self.translation_dim] - denoise_out[:,
                                                                                       self.size_dim:self.size_dim + self.translation_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_angle = ((target[:, self.size_dim + self.translation_dim:self.bbox_dim] - denoise_out[:,
                                                                                       self.size_dim + self.translation_dim:self.bbox_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_bbox = ((target[:, 0:self.bbox_dim] - denoise_out[:, 0:self.bbox_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        losses = ((target - denoise_out) ** 2).mean(dim=list(range(1, len(data_t.shape))))

        if self.loss_iou:
            loss_iou_valid, bbox_iou_valid = self.IoU_loss(data_t, timestep=t, pred_data=denoise_out,scene_ids=scene_ids)
        else:
            loss_iou_valid = torch.zeros(len(denoise_out)).to(data_t.device)
            bbox_iou_valid = torch.zeros(len(denoise_out)).to(data_t.device)

        return losses.mean() + loss_iou_valid.mean(), {
            'loss.bbox': loss_bbox.mean(),
            'loss.trans': loss_trans.mean(),
            'loss.size': loss_size.mean(),
            'loss.angle': loss_angle.mean(),
            'loss.liou': loss_iou_valid.mean(),
            'loss.bbox_iou': bbox_iou_valid.mean(),
        }

    def p_losses(self, denoise_fn, data_start, obj_embed, triples, t, condition_cross=None, scene_ids=None):
        """
        Training loss calculation
        """
        # make it compatible for 1D
        B, D = data_start.shape
        assert t.shape == torch.Size([B])

        # preprocess angle
        sincos = preprocess_angle2sincos(data_start[:,D-1:D])
        data_start = torch.concat((data_start[:,:D-1],sincos),dim=-1)

        noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise) # diffuse the bbox step by step

        if self.model_mean_type == 'eps':
            target = noise
        elif self.model_mean_type == 'x0':
            target = data_start
        else:
            raise NotImplementedError
        # predict the noise instead of x_start. seems to be weighted naturally like SNR
        denoise_out = denoise_fn(data_t, obj_embed, triples, t, condition_cross)
        assert data_t.shape == data_start.shape
        assert denoise_out.shape == data_start.shape
        loss, loss_dict = self.diffusion_loss(data_t, t, denoise_out, target, scene_ids)

        return loss, loss_dict


    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, condition, condition_cross, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b, condition=condition, condition_cross=condition_cross,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()
        


class DiffusionPoint(nn.Module):
    def __init__(self, denoise_net, config, conditioning_key=None, schedule_type='linear', beta_start=0.0001, beta_end=0.02, time_num=1000,
            loss_type='mse', model_mean_type='eps', model_var_type ='fixedsmall', loss_separate=False, loss_iou=False, iou_type = 'obb', train_stats_file=None):
          
        super(DiffusionPoint, self).__init__()
        
        betas = get_betas(schedule_type, beta_start, beta_end, time_num)

        
        self.diffusion = GaussianDiffusion(config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou, iou_type, train_stats_file)
        self.model = denoise_net


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, condition, condition_cross, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0,  condition, condition_cross, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, obj_embed, triples, t, condition_cross):
        B, D = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        # data = data.unsqueeze(1)
        if self.model.conditioning_key == 'concat':
            out = self.model(data, obj_embed, triples, t)
        elif self.model.conditioning_key == 'crossattn':
            out = self.model(data, obj_embed, triples, t, context=condition_cross)
        else:
            raise NotImplementedError
        # elif self.model.conditioning_key == 'hybrid':
        #     out = self.model(data, condition, t, context=condition_cross)
        out = out.squeeze(-1)

        assert out.shape == torch.Size([B, D])
        return out

    def get_loss_iter(self, obj_embed, preds, data, scene_ids=None, condition_cross=None):
        B, _ = data.shape

        unique_scenes, inv_idx = np.unique(scene_ids, return_inverse=True)
        t = torch.randint(0, self.diffusion.num_timesteps, size=unique_scenes.shape,
                          device=data.device)  # we want to have different t for each scene not each obj
        t = t[inv_idx]
        assert len(t) == B

        loss, loss_dict = self.diffusion.p_losses(self._denoise, data, obj_embed, triples=preds, t=t, condition_cross=condition_cross, scene_ids=scene_ids)
        assert t.shape == torch.Size([B])
        return loss, loss_dict
    

    def gen_samples(self, shape, device, condition=None, condition_cross=None, noise_fn=torch.randn,
                    clip_denoised=True, keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_samples_sg(self, shape, device, obj_embed, triples=None, condition=None, noise_fn=torch.randn,
                    clip_denoised=True, keep_running=False, scene_ids=None, floor_plan=None, room_outer_box=None, objectness=None):
        # [MODIFIED] Pass floor_plan and room_outer_box
        return self.diffusion.p_sample_loop_sg(self._denoise, shape=shape, device=device, obj_embed=obj_embed, triples=triples, condition=condition, scene_ids=scene_ids, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised, keep_running=keep_running, floor_plan=floor_plan, room_outer_box=room_outer_box, objectness=objectness)

    def get_latest_sampling_stats(self):
        return self.diffusion.latest_sampling_stats

    def gen_sample_traj(self, shape, device, freq, condition=None, condition_cross=None, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def gen_sample_traj_sg(self, shape, device, freq, condition=None, triples=None, condition_cross=None, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory_sg(self._denoise, shape=shape, device=device, condition=condition, triples=triples, condition_cross=condition_cross, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)