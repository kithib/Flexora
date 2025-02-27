import torch
import random
from torch.optim import Optimizer, Adam, AdamW
from torch.nn.functional import softmax
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
import time
import math
import warnings
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import itertools

# refer to https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py
@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector)."""
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
    Ei = (D * E).sum(dim=-1, keepdim=True)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits


def exact_conditional_gumbel(logits, D, k=1):
    """Same as conditional_gumbel but uses rejection sampling."""
    # Rejection sampling.
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(gumbel)
    return torch.stack(gumbels)


def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()


def gumbel_rao(logits, k=20, temp=1, I=None):
    """Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with gumbel-rao gradient.

    k: integer number of samples to use in the rao-blackwellization.
    1 sample reduces to straight-through gumbel-softmax.

    I: optional, categorical sample to use instead of drawing a new
    sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).

    """
    num_classes = logits.shape[-1]
    if I is None:
        I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)


# Optional [0, 1, 2]. 
    # 0: no print
    # 1: print the relative time whenever a parameter's grad is ready
    # 2: for debug usage only. Will set all the parameters trainable, print the grad ready time for each parameter. 
    #     In this case, all the grad except the "specified" trainable parameters will be set to None after being calculated.
BACKWARD_VERBOSE = 0
val_loss_list = []
class BlockOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on a specified block list."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        block_prefix_list: List[str],
        switch_block_every: int = 10,
        start_block: Optional[int] = None,
        switch_mode: str = "flora",
        active_modules: List[str] = [],
        include_embedding=False,
        include_lm_head=False,
        verbose: int = 1,
        log_fn = None,
    ):
        """
        Args:
            base_optimizer (Optimizer): The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list (List[List[str]]): The list of blocks of parameters to be updated.
            switch_block_every (int, optional): The number of optimization steps before switching to the next block. Defaults to 10.
            start_block (Optional[int], optional): The index of the block to start with. Defaults to None.
            switch_mode (str, optional): The mode for switching between different blocks of parameters. Defaults to "descending".
            active_modules (List[str]): The list of modules that are always active during optimization. Defaults to None.
            verbose (int, optional): The verbosity level for printing information during optimization. Defaults to 1.
            log_fn: A logging function for recording information during optimization. Defaults to None.
        """
        if block_prefix_list is None:
            block_prefix_list = self.infer_param_groups([n for n, _ in named_parameters_list], include_embedding, include_lm_head)

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)

        self.verbose = verbose
        self.switch_mode = switch_mode
        self.switch_block_every = switch_block_every
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.block_num = len(block_prefix_list)
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer
        
        # flora add
        self.hyper_param = torch.zeros(self.block_num, requires_grad=True)
        self.hyper_optimizer = Adam([self.hyper_param], lr=0.01)

        # detect if in lora mode or not
        self.lora_mode = False
        if any("lora" in n for n, _ in named_parameters_list):
            self.lora_mode = True
            print("LoRA mode detected. Will only train the lora parameters.")
            
        if any(isinstance(p, torch.FloatTensor) for _, p in named_parameters_list):
            warnings.warn("BAdam expect model to be loaded in fp16 precision while detect fp32 weight. \
                This will cause additional memory usage and lose the benefit of mixed precision training.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)
        
        if BACKWARD_VERBOSE:
            self.record_mark = True
            self.ordered_named_params = []
            self.param_num = len(named_parameters_list)
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.test_hook(n))

        self.update_trainable_params()

        if BACKWARD_VERBOSE == 2:
            for name, param in self.named_parameters_list:
                param.requires_grad_(True)


    
    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names, include_embedding, include_lm_head):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        import re
        
        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)
        
        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)
        
        return block_prefix_list
                
    def test_hook(self, name):
        """hook used for recording the time of gradient calculation, see comments on BACKWARD_VERBOSE for more details."""
        
        def func(x):
            if self.record_mark:
                self.backward_start_time = time.time()          
                self.record_mark = False
                relative_time = 0.
            else:
                relative_time = time.time() - self.backward_start_time
            if any(p_name in name for p_name in self.active_param_prefixs):
                print(f"param: {name:<50} relative time: {relative_time}")
            
            iterator = self.named_parameters_list
                
            for n, p in iterator:
                
                if p.requires_grad and p.grad is not None:
                    print("parameter name: ", n, "relative time", time.time() - self.backward_start_time)
                    
                    if (not any(p_name in n for p_name in self.active_param_prefixs)) and \
                        BACKWARD_VERBOSE == 2:
                        p.grad = None
                    
                    if len(self.ordered_named_params) < self.param_num:
                        self.ordered_named_params.append((n, p))
                    # break since for each step only one parameter's grad is updated
                    break
            return x
        
        return func

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]
            
    def step(self, *args, **kwargs) -> None:
        self.record_mark = True
        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()
        self.global_step += 1
        
        # flora add
        self.hyper_step()

        torch.cuda.empty_cache()
        if (self.global_step + 1) % self.switch_block_every == 0:
            self.update_trainable_params()
    
    ############################ flora begin ################################
    
    def selected_layer(self):
        layers = torch.nonzero(self.hyper_param > 0.0, as_tuple=False)
        return [idx.item() for idx in layers]
    
    def inject_hyper_param(self, model_train=False):
        import re
        from functools import partial
        
        def wgt_lora(module, input, weight): # equivalent to multiply lora weights
            wgt_input = (input[0] * weight, )
            return wgt_input
        
        self.hooks = []
        for name, module in self.model.named_modules():
            out = re.search(r'layers\.(\d+)', name)
            if out is not None and "lora" in name: # layer-wise change for lora
                block_idx = int(out.group(1).split(".")[-1])
                # weight = self.sample_block_onehot[block_idx][-1]
                weight = self.block_num * softmax(self.hyper_param, dim=-1)[block_idx]
                hook = module.register_forward_pre_hook(partial(
                    wgt_lora, weight=weight.detach() if model_train else weight
                ))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    # flora add
    def register_model_and_trainer(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.val_iter = itertools.cycle(iter(trainer.get_eval_dataloader()))
    
    # flora add
    def hyper_step(self, n=8):
        self.model.eval() # avoid grad to model params
        self.remove_hooks() # remove existing hooks during model training
        
        self.inject_hyper_param(model_train=False)
        self.hyper_optimizer.zero_grad()
        
        for _ in range(n):
            val_inputs = next(self.val_iter)
            val_inputs = self.trainer._prepare_inputs(val_inputs)
            
            label_loss = self.trainer.compute_loss(self.model, val_inputs)
            loss = label_loss 
            loss.backward(retain_graph=True)  # 反向传播，保留计算图
            del val_inputs

        self.hyper_optimizer.step()  # 更新超参数
        torch.cuda.empty_cache()
    
        if (self.global_step + 1) % 1 == 0:
            # print("reg loss:", reg_loss.item())
            print("val loss:", loss.item())
            val_loss_list.append(loss.item())
            print("current_block_idx:", self.current_block_idx)
            print("current selected layer:", self.selected_layer())
            print("current hyper-param:", self.hyper_param)
            # val_loss_list.append(loss.item())
    
        self.model.train()
        self.remove_hooks()
        self.inject_hyper_param(model_train=True)
    
    ############################ flora end ################################
        
    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        grad_norm = 0.0
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def update_trainable_params(self, verbose: Optional[int] = None) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.

        Args:
            verbose (Optional[int], optional): The verbosity level for printing information. Defaults to None.
        """
        if verbose is None:
            verbose = self.verbose

        self._update_active_block()
        selected_prefixes = [
            self.block_prefix_list[idx][0] for idx in self.current_block_idx if self.block_prefix_list[idx][0] is not None
        ]
        self.active_param_prefixs = selected_prefixes + self.active_modules
        
        # Make sure there are trainable parameters in the current block when using lora
        while self.lora_mode:
            active_param_names = [n for n, _ in self.named_parameters_list if any(p in n for p in self.active_param_prefixs)]
            if all("lora" not in n for n in active_param_names):
                print(f"In LoRA mode but no lora parameters in the current block with prefix: {self.active_param_prefixs}. Switching to the next block.")
                self._update_active_block()
                selected_prefixes = [
                    self.block_prefix_list[idx][0] for idx in self.current_block_idx if self.block_prefix_list[idx][0] is not None
                ]
                self.active_param_prefixs = selected_prefixes + self.active_modules
                continue
            break
        
        if verbose >= 1:
            print("Parameters with the following prefix will be trainable:", self.active_param_prefixs)

        # Reset parameters to be optimized
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]

        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                if self.lora_mode and "lora" not in name:
                    continue
                param.requires_grad_(True)
                param_hp = param.clone().float().detach().to(param.device)
                param_hp.requires_grad = True
                
                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)
                
                if verbose >= 2:
                    print(name)

        self.base_optimizer.param_groups = active_param_groups
        
        import gc
        gc.collect()
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})
        # self._update_active_block()

    def _update_active_block(self):
        # flora modified
        # max_n_index = torch.argsort(self.hyper_param[:,-1].detach(), descending=True)[:5]
        # self.sample_block_onehot = gumbel_rao(self.hyper_param) # [max_n_index,:]
        # self.current_block_idx = torch.nonzero(self.sample_block_onehot[:,-1].detach(), as_tuple=True)[0] #.tolist()
        self.current_block_idx = [i for i in range(self.block_num)]

    
# For torch>=2.1, `_foreach_norm` is used when implementing `clip_grad_norm_`, which doesn't support sparse tensor yet.
# We can temporarily fix this issue by using the older torch version's implementation:
    # self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
def clip_grad_norm_for_sparse_tensor(self, parameters, max_norm, norm_type=2):
    """
    Modification of the accelerator.clip_grad_norm_ to enable gradient clipping for sparse tensor.
    Used for torch version >= 2.1
    """
    from accelerate.utils import DistributedType
    from torch import inf

    if self.distributed_type == DistributedType.FSDP:
        self.unscale_gradients()
        parameters = [p for p in parameters]
        for model in self._models:
            if parameters == [p for p in model.parameters()]:
                return model.clip_grad_norm_(max_norm, norm_type)
    elif self.distributed_type == DistributedType.DEEPSPEED:
        # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
        # We cannot return the gradient norm because DeepSpeed does it.
        return None
    self.unscale_gradients()
    
    def clip_func_(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
        r""" torch 1.13 version clip_grad_norm_, works well with sparse tensor.
        Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        device = grads[0].device
        if norm_type == inf:
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        return total_norm
    
    return clip_func_(parameters, max_norm, norm_type=norm_type)
