"""
Here is an original implementation of ADOPT. 
Source: https://github.com/iShohei220/adopt
"""

import torch


def exists(val):
    return val is not None


from typing import Callable, Optional, Tuple

import torch


def adopt_clip_fn(step: int) -> float:
    return step ** 0.25


class ADOPT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[int], float]] = adopt_clip_fn,
        weight_decay: float = 0.0,
        decouple: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            clip_lambda=clip_lambda,
            step=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['step'] += 1
            step = group['step']
            beta1, beta2 = group["betas"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                state = self.state[p]

                if len(state) ==0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if step == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                    continue

                if group["weight_decay"] != 0:
                    if group["decouple"]:
                        p.data.mul_(1 - lr * group["weight_decay"])
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                denom = torch.clamp(exp_avg_sq.sqrt(), group["eps"])
                normed_grad = grad.div(denom)

                if group["clip_lambda"] is not None:
                    clip = group["clip_lambda"](step)
                    normed_grad.clamp_(-clip, clip)

                exp_avg.lerp_(normed_grad, 1 - beta1)
                p.data.add_(exp_avg, alpha=-lr)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        return loss
