from abc import ABC, abstractmethod

import numpy as np
import torch

from util.img_utils import dynamic_thresholding



# ====================
# Model Mean Processor
# ====================

class MeanProcessor(ABC):
    """Predict x_start and calculate mean value"""
    @abstractmethod
    def __init__(self, betas, dynamic_threshold, clip_denoised):
        self.dynamic_threshold = dynamic_threshold
        self.clip_denoised = clip_denoised

    @abstractmethod
    def get_mean_and_xstart(self, x, t, model_output):
        pass

    def process_xstart(self, x):
        if self.dynamic_threshold:
            x = dynamic_thresholding(x, s=0.95)
        if self.clip_denoised:
            x = x.clamp(-1, 1)
        return x

class EpsilonXMeanProcessor(MeanProcessor):
    def __init__(self, betas, dynamic_threshold, clip_denoised):
        super().__init__(betas, dynamic_threshold, clip_denoised)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0-alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)


    def q_posterior_mean(self, x_start, x_t, t):
        """
        Compute the mean of the diffusion posteriro:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_start + coef2 * x_t
    
    def predict_xstart(self, x_t, t, eps):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, eps)
        # print('using this')
        return coef1 * x_t - coef2 * eps

    def get_mean_and_xstart(self, x, t, model_output):
        pred_xstart = self.process_xstart(self.predict_xstart(x, t, model_output))
        mean = self.q_posterior_mean(pred_xstart, x, t)

        return mean, pred_xstart

# =========================
# Model Variance Processor
# =========================

class VarianceProcessor(ABC):
    @abstractmethod
    def __init__(self, betas):
        pass

    @abstractmethod
    def get_variance(self, x, t):
        pass

# @register_var_processor(name='learned')
class LearnedVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        pass

    def get_variance(self, x, t):
        model_log_variance = x
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance

# @register_var_processor(name='learned_range')
class LearnedRangeVarianceProcessor(VarianceProcessor):
    def __init__(self, betas):
        self.betas = betas

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

    def get_variance(self, x, t):
        model_var_values = x
        min_log = self.posterior_log_variance_clipped
        max_log = np.log(self.betas)

        min_log = extract_and_expand(min_log, t, x)
        max_log = extract_and_expand(max_log, t, x)

        # The model_var_values is [-1, 1] for [min_var, max_var]
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1-frac) * min_log
        model_variance = torch.exp(model_log_variance)
        return model_log_variance

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)
