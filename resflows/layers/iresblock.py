import math
import numpy as np
import torch
import torch.nn as nn

import logging

logger = logging.getLogger()

__all__ = ['iResBlock']


class iResBlock(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=False,
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        nn.Module.__init__(self)
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, logpx=None):
        if logpx is None:
            y = x + self.nnet(x)
            return y
        else:
            g, logdetgrad = self._mylogdetgrad(x)
            return x + g, logpx - logdetgrad

    def inverse(self, y, logpy=None):
        x = self._my_inverse_fixed_point(y)
        if logpy is None:
            return x
        else:
            return x, logpy + self._mylogdetgrad(x)[1]

    def _my_inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5, m=5):
        return self._inverse_anderson_acceleration(y, atol, rtol)

    def _inverse_anderson_acceleration(self, y, atol=1e-5, rtol=1e-5, m=5):
        def fixed_point_map(x):
            return y - self.nnet(x)

        x = y - self.nnet(y) 
        tol = atol + y.abs() * rtol

        G_history = []
        F_history = []

        for i in range(100):
            x_prev = x.clone()
            g = fixed_point_map(x)
            f = g - x  # 잔차

            if torch.all((x - x_prev)**2 / tol < 1):
                break

            if len(F_history) > 0 and len(F_history) <= m:
                F_mat = torch.stack(F_history, dim=-1)  # [batch, dim, history_len]
                Delta_F = F_mat[:, :, -1:] - F_mat[:, :, :-1] if F_mat.shape[-1] > 1 else F_mat

                if Delta_F.shape[-1] > 0:
                    try:
                        batch_size = f.shape[0]
                        alpha_list = []

                        for b in range(batch_size):
                            Delta_F_b = Delta_F[b]  # [dim, history_len-1]
                            f_b = f[b].unsqueeze(-1)  # [dim, 1]

                            # QR 분해를 사용한 안정적인 해결
                            Q, R = torch.linalg.qr(Delta_F_b)
                            alpha_b = torch.linalg.solve_triangular(R, Q.T @ f_b, upper=True)
                            alpha_list.append(alpha_b.squeeze(-1))

                        alpha = torch.stack(alpha_list, dim=0)  # [batch, history_len-1]

                        G_mat = torch.stack(G_history, dim=-1)
                        if G_mat.shape[-1] > 1:
                            Delta_G = G_mat[:, :, -1:] - G_mat[:, :, :-1]
                            correction = torch.sum(Delta_G * alpha.unsqueeze(1), dim=-1)
                            x = g - correction
                        else:
                            x = g

                    except:
                        x = g
                else:
                    x = g
            else:
                x = g

            G_history.append(g.clone())
            F_history.append(f.clone())

            if len(G_history) > m:
                G_history.pop(0)
                F_history.pop(0)

        if i >= 99:
            logger.warning(f'Anderson acceleration: Iterations exceeded 100 for inverse.')

        return x


    def _mylogdetgrad(self,x):
        with torch.enable_grad():
            if not self.training:
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).view(-1, 1)
            
            p = torch.sigmoid(self.geom_p).item()
            sampling = lambda m: np.random.geometric(p,m)
            r_cdf = lambda k, offset: 1 if k<= offset else (1-p)**max(k-offset-1,0)

            if self.n_power_series is None:
                n_samples  = sampling(self.n_samples)
                n_power_series = max(n_samples) + self.n_exact_terms
                coeff = lambda k: 1 / r_cdf(k, self.n_exact_terms) * \
                    sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
            else:
                n_power_series = self.n_power_series
                coeff = lambda k: 1
            
            estimator = my_neumann_logdet_estimator
            rand_v = torch.randn_like(x)

            if self.training and self.grad_in_forward:
                g, logdetgrad = mem_eff_wrapper(
                    estimator, self.nnet, x, n_power_series, rand_v, coeff, self.training
                )
            else:
                x = x.requires_grad_(True)
                g = self.nnet(x)
                logdetgrad = estimator(g, x, n_power_series, rand_v, coeff, self.training)

            return g, logdetgrad.view(-1,1)


    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force
        )


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


#####################
# Logdet Estimators
#####################
class MyMemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator, gnet, x, n_power_series, vareps, coeff, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator(g, x, n_power_series, vareps, coeff, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, rand_v, coeff_fn, training):
    vjp = rand_v
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * rand_v.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def my_neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad



def mem_eff_wrapper(estimator, gnet, x, n_power_series, rand_v, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MyMemoryEfficientLogDetEstimator.apply(
        estimator, gnet, x, n_power_series, rand_v, coeff_fn, training, *list(gnet.parameters())
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).view(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])
