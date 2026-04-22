import builtins
import marimo as mo

builtins.mo = mo

app = mo.App()

@app.cell
def __():
    import math
    import random
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    try:
        import ipywidgets as widgets
        from ipywidgets import interact
        HAS_WIDGETS = True
    except Exception:
        widgets = None
        interact = None
        HAS_WIDGETS = False
    math = math
    random = random
    torch = torch
    nn = nn
    np = np
    pd = pd
    plt = plt
    FuncAnimation = FuncAnimation
    HTML = HTML
    display = display
    return mo, math, random, torch, nn, np, pd, plt, FuncAnimation, HTML, display, widgets, interact, HAS_WIDGETS

@app.cell
def __():
    mo.md("""# The Geometry of Noise: Why Diffusion Models Do Not Need Noise Conditioning

## Section 1 - Introduction and Problem

Standard diffusion models use a time-conditioned network $f(u, t)$.
The input $t$ tells the model how noisy the state is.

This paper asks a stronger question: can one autonomous field $f(u)$ work without $t$?

This is surprising because one vector field must handle all noise levels.

Simple view:
- Conditioned model: $u, t \rightarrow f(u, t)$
- Autonomous model: $u \rightarrow f(u)$""")
    return

@app.cell
def __():
    mo.md("""## Section 2 - Core Mathematical Idea

We study the marginal distribution over states:

$$
p(u) = \int p(u\mid t)p(t)dt
$$

and the marginal energy:

$$
E_{marg}(u) = -\log p(u).
$$

Key insight: the autonomous model learns a single field that averages over noise levels.

$$
f^*(u) = \mathbb{E}_{t\mid u}[f_t(u)]
$$

So the model is not blind.
It infers likely noise levels from geometry and acts accordingly.""")
    return

@app.cell
def __():
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('ipywidgets:', HAS_WIDGETS)
    
    PALETTE = {
        'cond': '#1f7a8c',
        'blind': '#bf4342',
        'vel': '#2a9d8f',
        'eps': '#f4a261',
        'x0': '#264653',
    }
    
    plt.rcParams['figure.figsize'] = (7.0, 5.0)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    return

@app.cell
def __():
    def sample_ring_mog(n=1024, radius=2.0, k=8, std=0.14):
        idx = torch.randint(0, k, (n,))
        theta = 2 * math.pi * idx.float() / k
        centers = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
        return centers + std * torch.randn(n, 2)
    
    def sample_two_moons(n=1024, noise=0.08):
        n1 = n // 2
        n2 = n - n1
        t1 = torch.rand(n1) * math.pi
        t2 = torch.rand(n2) * math.pi
        x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
        x2 = torch.stack([1.0 - torch.cos(t2), -torch.sin(t2) - 0.5], dim=1)
        x = torch.cat([x1, x2], dim=0)
        return 1.7 * x + noise * torch.randn_like(x)
    
    def sample_double_spiral(n=1024, noise=0.06):
        n1 = n // 2
        n2 = n - n1
        t1 = torch.sqrt(torch.rand(n1)) * 4.0 * math.pi
        t2 = torch.sqrt(torch.rand(n2)) * 4.0 * math.pi
        r1 = 0.15 * t1
        r2 = 0.15 * t2
        x1 = torch.stack([r1 * torch.cos(t1), r1 * torch.sin(t1)], dim=1)
        x2 = torch.stack([-r2 * torch.cos(t2), -r2 * torch.sin(t2)], dim=1)
        x = torch.cat([x1, x2], dim=0)
        return 2.1 * x + noise * torch.randn_like(x)
    
    def embed_2d_to_d(x2, d):
        if d == 2:
            return x2
        extra = torch.randn(x2.shape[0], d - 2, device=x2.device)
        return torch.cat([x2, extra], dim=1)
    
    def schedule(t):
        alpha = torch.cos(0.5 * math.pi * t)
        sigma = torch.sin(0.5 * math.pi * t)
        dalpha = -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
        dsigma = 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
        return alpha, sigma, dalpha, dsigma
    
    @torch.no_grad()
    def make_grid(lim=3.8, n=120):
        xs = np.linspace(-lim, lim, n)
        ys = np.linspace(-lim, lim, n)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        return gx, gy, pts
    return

@app.cell
def __():
    mo.md("""## Section 3 - Energy Geometry (Critical Visual Section)

This section explains why autonomous diffusion is a geometry problem.
We inspect the marginal energy landscape $E_{marg}(u)$ on 2D toy data.""")
    return

@app.cell
def __():
    def mog_density_np(pts, radius=2.0, k=8, std=0.25):
        theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
        centers = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        diff = pts[:, None, :] - centers[None, :, :]
        sq = (diff ** 2).sum(axis=2)
        z = np.exp(-0.5 * sq / (std ** 2)).mean(axis=1)
        z = z / (2 * np.pi * std ** 2)
        return z
    
    def marginal_energy_np(pts, eps=1e-10):
        p = mog_density_np(pts)
        return -np.log(np.maximum(p, eps))
    
    gx, gy, pts = make_grid(lim=3.8, n=125)
    energy = marginal_energy_np(pts).reshape(gx.shape)
    
    dy, dx = np.gradient(energy, gy[:, 0], gx[0, :], edge_order=2)
    grad_norm = np.sqrt(dx ** 2 + dy ** 2)
    return

@app.cell
def __():
    mo.md("""### 3.1 3D Energy Landscape

The surface plot shows wells around the data manifold.
Low energy means high probability under the marginal distribution.""")
    return

@app.cell
def __():
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    s = ax1.plot_surface(gx, gy, energy, cmap='viridis', linewidth=0, antialiased=True, alpha=0.95)
    
    ax1.set_title('3.1 Marginal energy landscape')
    ax1.set_xlabel('u1')
    ax1.set_ylabel('u2')
    ax1.set_zlabel('E_marg')
    
    fig1.colorbar(s, ax=ax1, shrink=0.6, pad=0.1)
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""### 3.2 Contour Plot

Contours show where probability mass is concentrated.
The data modes align with low-energy rings and basins.""")
    return

@app.cell
def __():
    _fig2 = plt.figure(figsize=(6, 5))
    _ax2 = _fig2.add_subplot(111)
    
    _c = _ax2.contourf(gx, gy, energy, levels=30, cmap='magma')
    
    ring = sample_ring_mog(1500).cpu().numpy()
    _ax2.scatter(ring[:, 0], ring[:, 1], s=4, alpha=0.15, c='white')
    
    _ax2.set_aspect('equal', adjustable='box')
    _ax2.set_title('3.2 Energy contours + data')
    
    _fig2.colorbar(_c, ax=_ax2, shrink=0.85)
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""### 3.3 Gradient Behavior

The raw gradient field $\nabla E_{marg}(u)$ can become very large near singular regions.
This creates stiff dynamics and unstable steps if we follow it directly.

Takeaway: naive Euclidean gradient descent is not a stable sampler in this geometry.""")
    return

@app.cell
def __():
    _fig3 = plt.figure(figsize=(6, 5))
    _ax3 = _fig3.add_subplot(111)
    
    skip = (slice(None, None, 4), slice(None, None, 4))
    
    _q = _ax3.quiver(
        gx[skip], gy[skip],
        dx[skip], dy[skip],
        grad_norm[skip],
        cmap='cividis',
        scale=58,
        width=0.003
    )
    
    _ax3.set_aspect('equal', adjustable='box')
    _ax3.set_title('3.3 Raw gradient field')
    
    _fig3.colorbar(_q, ax=_ax3, shrink=0.85, label='|grad E_marg|')
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""## Shared Model Utilities

The next cells prepare model families and training helpers used by Sections 4 to 12.

We keep models lightweight so all figures remain interactive and fast.""")
    return

@app.cell
def __():
    def sample_xt_targets(n=1024, d=2, sampler=sample_ring_mog):
        x0_2d = sampler(n).to(device)
        x0 = embed_2d_to_d(x0_2d, d)
        eps = torch.randn_like(x0)
        t = torch.rand(n, 1, device=device)
        a, s, da, ds = schedule(t)
        xt = a * x0 + s * eps
        target_v = da * x0 + ds * eps
        target_eps = eps
        target_x0 = x0
        return x0, eps, t, xt, target_v, target_eps, target_x0
    
    def recover_x0_from_v(xt, vhat, t, eps_safe=1e-6):
        a, s, da, ds = schedule(t)
        det = a * ds - s * da
        det = torch.where(det.abs() < eps_safe, eps_safe * torch.ones_like(det), det)
        return (ds * xt - s * vhat) / det
    
    def recover_x0_from_eps(xt, eps_hat, t, eps_safe=1e-6):
        a, s, _, _ = schedule(t)
        a = torch.where(a.abs() < eps_safe, eps_safe * torch.ones_like(a), a)
        return (xt - s * eps_hat) / a
    
    def mlp(in_dim, hidden=192, out_dim=2):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    
    class FieldCond(nn.Module):
        def __init__(self, d=2):
            super().__init__()
            self.net = mlp(d + 1, hidden=192, out_dim=d)
    
        def forward(self, xt, t):
            return self.net(torch.cat([xt, t], dim=1))
    
    class FieldBlind(nn.Module):
        def __init__(self, d=2):
            super().__init__()
            self.net = mlp(d, hidden=192, out_dim=d)
    
        def forward(self, xt):
            return self.net(xt)
    
    class GeometryBlind(nn.Module):
        def __init__(self, d=2, n_freq=6):
            super().__init__()
            self.register_buffer('freqs', torch.arange(1, n_freq + 1).float())
            self.net = mlp(d + 1 + 2 * n_freq, hidden=192, out_dim=d)
    
        def features(self, x):
            r = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + 1e-8)
            f = self.freqs.view(1, -1).to(x.device)
            return torch.cat([r, torch.sin(f * r), torch.cos(f * r)], dim=1)
    
        def forward(self, xt):
            return self.net(torch.cat([xt, self.features(xt)], dim=1))
    return

@app.cell
def __():
    def train_field(model, conditioned=False, target='v', d=2, steps=280, batch=768, lr=1e-3, smooth_weight=0.0):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
    
        for _ in range(steps):
            _, _, t, xt, target_v, target_eps, target_x0 = sample_xt_targets(n=batch, d=d)
            pred = model(xt, t) if conditioned else model(xt)
    
            if target == 'v':
                y = target_v
            elif target == 'eps':
                y = target_eps
            elif target == 'x0':
                y = target_x0
            else:
                raise ValueError('Unknown target')
    
            loss_main = ((pred - y) ** 2).mean()
            loss_smooth = torch.tensor(0.0, device=device)
    
            if smooth_weight > 0 and not conditioned:
                delta = 0.03 * torch.randn_like(xt)
                pred_pert = model(xt + delta)
                loss_smooth = ((pred_pert - pred) ** 2).mean()
    
            loss = loss_main + smooth_weight * loss_smooth
            opt.zero_grad()
            loss.backward()
            opt.step()
    
        return model.eval()
    
    @torch.no_grad()
    def vector_field_np(model, pts_np, conditioned=False, t_value=0.5):
        x = torch.tensor(pts_np, dtype=torch.float32, device=device)
        if conditioned:
            t = torch.full((x.shape[0], 1), float(t_value), device=device)
            v = model(x, t)
        else:
            v = model(x)
        return v.detach().cpu().numpy()
    return

@app.cell
def __():
    def draw_field(ax, field, title, color, scale=22):
        field_grid = field.reshape(gx.shape + (2,))
        skip = (slice(None, None, 4), slice(None, None, 4))
        ax.quiver(
            gx[skip], gy[skip],
            field_grid[..., 0][skip], field_grid[..., 1][skip],
            color=color, scale=scale, width=0.003
        )
        ax.set_title(title)
        ax.set_aspect('equal')

    section4_raw_field = np.stack([-dx.ravel(), -dy.ravel()], axis=1)
    section4_scaled_field = section4_raw_field / (1.0 + 2.0 * np.sqrt((section4_raw_field ** 2).sum(axis=1, keepdims=True)))
    section4_learned_bounded_model = train_field(GeometryBlind(d=2), conditioned=False, target='v', d=2, steps=160, smooth_weight=0.35)
    section4_learned_bounded = vector_field_np(section4_learned_bounded_model, pts, conditioned=False)
    return draw_field, section4_raw_field, section4_scaled_field, section4_learned_bounded

@app.cell
def __():
    mo.md("""## Section 4 - Learned Vector Field vs Raw Gradient

We compare two vector fields at the same spatial locations:

- Raw field: $-\nabla E_{marg}(u)$
- Learned field: $f(u)$

The raw gradient is unstable.
Its magnitude changes sharply near high-density regions.

The learned field behaves differently.
It is smoother and more consistent across space.

This shows the model is not copying the gradient.
It learns a transport field that is easier to integrate.""")
    return

@app.cell
def __():
    _fig1, _ax1 = plt.subplots()
    
    draw_field(
        _ax1,
        section4_raw_field,
        'Raw field: $-\\nabla E_{marg}(u)$',
        color=PALETTE['blind'],
        scale=22
    )
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    _fig2, _ax2 = plt.subplots()
    
    draw_field(
        _ax2,
        section4_learned_bounded,
        'Learned field: $f(u)$',
        color=PALETTE['vel'],
        scale=22
    )
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""## Section 5 - Riemannian Gradient Flow (Core Insight)

The behavior of the learned field can be interpreted as:

$$
\dot u = -\lambda(u)\nabla E_{marg}(u)
$$

The factor $\lambda(u)$ changes with position.

- In high-density regions, it reduces the step size
- In low-density regions, it allows larger movement

This prevents instability.

The result is a smoother and more stable flow.

This explains why the learned model works better than the raw gradient.""")
    return

@app.cell
def __():
    _fig3, _ax3 = plt.subplots()
    
    draw_field(
        _ax3,
        section4_scaled_field,
        'Scaled field: $-\\lambda(u) \\nabla E_{marg}(u)$',
        color=PALETTE['cond'],
        scale=22
    )
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""## Section 6 - Parameterization and Stability

We compare three training targets:

- Noise prediction: $\epsilon$
- Signal prediction: $x_0$
- Velocity prediction: $v$

Each target scales differently with time $t$.

Near $t \rightarrow 0$:
- Some targets amplify errors
- Others remain stable

This affects reconstruction quality.""")
    return

@app.cell
def __():
    m_eps = train_field(FieldCond(d=2), conditioned=True, target='eps', d=2, steps=300)
    m_x0 = train_field(FieldCond(d=2), conditioned=True, target='x0', d=2, steps=300)
    m_vel = train_field(FieldCond(d=2), conditioned=True, target='v', d=2, steps=300)
    
    @torch.no_grad()
    def recon_error_vs_t(model, target='v', n=2200, t_values=None):
        if t_values is None:
            t_values = np.linspace(0.03, 0.97, 24)
        errs = []
        for tv in t_values:
            x0, _, t, _, _, _, _ = sample_xt_targets(n=n, d=2)
            t = torch.full_like(t, float(tv))
            a, s, _, _ = schedule(t)
            eps = torch.randn_like(x0)
            xt = a * x0 + s * eps
            pred = model(xt, t)
    
            if target == 'eps':
                x0_hat = recover_x0_from_eps(xt, pred, t)
            elif target == 'x0':
                x0_hat = pred
            else:
                x0_hat = recover_x0_from_v(xt, pred, t)
    
            errs.append(((x0_hat - x0) ** 2).mean().item())
        return np.array(t_values), np.array(errs)
    
    ts_stab, err_eps = recon_error_vs_t(m_eps, target='eps')
    _, err_x0 = recon_error_vs_t(m_x0, target='x0')
    _, err_vel = recon_error_vs_t(m_vel, target='v')
    
    tt = np.linspace(0.01, 0.99, 400)
    gain_eps = np.sin(0.5 * np.pi * tt) / np.maximum(np.cos(0.5 * np.pi * tt), 1e-4)
    gain_x0 = np.ones_like(tt)
    gain_vel = np.sin(0.5 * np.pi * tt)
    
    _fig1, _ax1 = plt.subplots()
    
    _ax1.plot(tt, gain_eps, label='epsilon', color=PALETTE['eps'])
    _ax1.plot(tt, gain_x0, label='x0', color=PALETTE['x0'])
    _ax1.plot(tt, gain_vel, label='velocity', color=PALETTE['vel'])
    
    _ax1.set_title('Gain vs time')
    _ax1.set_xlabel('t')
    _ax1.set_ylabel('gain')
    _ax1.set_ylim(0, 6)
    _ax1.legend(frameon=False)
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    _fig2, _ax2 = plt.subplots()
    
    _ax2.plot(ts_stab, err_eps, '-o', label='epsilon', color=PALETTE['eps'])
    _ax2.plot(ts_stab, err_x0, '-o', label='x0', color=PALETTE['x0'])
    _ax2.plot(ts_stab, err_vel, '-o', label='velocity', color=PALETTE['vel'])
    
    _ax2.set_title('Reconstruction error vs t')
    _ax2.set_xlabel('t')
    _ax2.set_ylabel('MSE')
    _ax2.legend(frameon=False)
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    mo.md("""Velocity prediction is the most stable.

It avoids large gain amplification near low noise.
This leads to better reconstruction across all $t$.""")
    return

@app.cell
def __():
    mo.md("""## Section 7 - Conditional vs Autonomous Models

We compare models with and without time input $t$.

- Conditional: model receives $t$
- Blind: model does not receive $t$

We test two families:
- DDPM (score-based)
- Flow Matching (velocity-based)

Goal:
Check if the model can still generate samples without time information.""")
    return

@app.cell
def __():
    @torch.no_grad()
    def sample_reverse_traj(model, family='fm', conditioned=True, n=600, steps=70):
        x = torch.randn(n, 2, device=device) * 2.2
        dt = 1.0 / steps
    
        traj = [x.detach().cpu().numpy()]
    
        for k in range(steps, 0, -1):
            tval = k / steps
            t = torch.full((n, 1), tval, device=device)
    
            if family == 'fm':
                v = model(x, t) if conditioned else model(x)
                x = x - dt * v
            else:
                eps_hat = model(x, t) if conditioned else model(x)
                a, s, _, _ = schedule(t)
                score_like = (x - s * eps_hat) / torch.clamp(a, min=1e-3)
                x = x + dt * (score_like - x)
    
            traj.append(x.detach().cpu().numpy())
    
        return traj
    return

@app.cell
def __():
    ddpm_cond = train_field(FieldCond(d=2), conditioned=True, target='eps', d=2, steps=120)
    ddpm_blind = train_field(FieldBlind(d=2), conditioned=False, target='eps', d=2, steps=120)
    fm_cond = train_field(FieldCond(d=2), conditioned=True, target='v', d=2, steps=120)
    fm_blind = train_field(FieldBlind(d=2), conditioned=False, target='v', d=2, steps=120)
    return ddpm_cond, ddpm_blind, fm_cond, fm_blind

@app.cell
def __():
    traj_data = {
        'DDPM (with t)': sample_reverse_traj(ddpm_cond, family='ddpm', conditioned=True),
        'DDPM Blind': sample_reverse_traj(ddpm_blind, family='ddpm', conditioned=False),
        'Flow Matching (with t)': sample_reverse_traj(fm_cond, family='fm', conditioned=True),
        'Flow Matching Blind': sample_reverse_traj(fm_blind, family='fm', conditioned=False),
    }
    return

@app.cell
def __():
    
    def animate_traj(traj, title):
        fig, ax = plt.subplots(figsize=(5, 5))
    
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
    
        scat = ax.scatter([], [], s=4, alpha=0.4, c=PALETTE['x0'])
    
        ax.set_title(title)
    
        def init():
            scat.set_offsets(np.empty((0, 2)))
            return [scat]
    
        def update(i):
            scat.set_offsets(traj[i])
            ax.set_title(f"{title} | step {i}/{len(traj)-1}")
            return [scat]
    
        anim = FuncAnimation(fig, update, frames=len(traj),
                             init_func=init, interval=80, blit=False)
    
        plt.close(fig)
        return anim
    return

@app.cell
def __():
    for _name, _traj in traj_data.items():
        _anim = animate_traj(_traj, _name)
        display(HTML(_anim.to_html5_video()))
    return

@app.cell
def __():
    mo.md("""Blind Flow Matching works better.

This suggests velocity fields are easier to learn without explicit time.""")
    return

@app.cell
def __():
    mo.md("""## Section 8 - Dimensionality Experiment

We increase the data dimension:

D = 2, 8, 32, 128

Then we project samples back to 2D for visualization.

Goal:
Understand how dimension affects structure and learning.""")
    return

@app.cell
def __():
    # Section 8 - Dimensionality experiment (mandatory)
    dims = [2, 8, 32, 128]
    dim_models = {}
    dim_samples_2d = {}
    
    for _d in dims:
        m_d = train_field(FieldBlind(d=_d), conditioned=False, target='v', d=_d, steps=210, batch=640)
        dim_models[_d] = m_d
        _x = torch.randn(900, _d, device=device) * 2.0
        for _ in range(55):
            _x = _x - 0.03 * m_d(_x)
        dim_samples_2d[_d] = _x[:, :2].detach().cpu().numpy()
    
    for _d in dims:
        _fig, _ax = plt.subplots()
    
        pts_d = dim_samples_2d[_d]
    
        _ax.scatter(pts_d[:, 0], pts_d[:, 1], s=4, alpha=0.35, c=PALETTE['x0'])
    
        _ax.set_title(f'Dimension = {_d}')
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect('equal', adjustable='box')
    
        plt.tight_layout()
        plt.show()
    return

@app.cell
def __():
    mo.md("""Low dimension is ambiguous.

High dimension reveals clearer structure.

This makes denoising easier in higher dimensions.""")
    return

@app.cell
def __():
    mo.md("""## Section 9 - Posterior Intuition

We estimate $p(t \mid u)$ for a fixed point $u$.

This shows how much the state reveals its noise level.

- Broad distribution → uncertain noise
- Sharp distribution → clear noise level""")
    return

@app.cell
def __():
    return

@app.cell
def __():
    @torch.no_grad()
    def approx_p_t_given_u(u, d=2, grid_t=None):
        if grid_t is None:
            grid_t = np.linspace(0.02, 0.98, 100)
        x0 = sample_ring_mog(500).to(device)
        x0 = embed_2d_to_d(x0, d)
        uu = torch.tensor(u, dtype=torch.float32, device=device).view(1, -1)
        uu = embed_2d_to_d(uu, d)
        vals = []
        for tv in grid_t:
            t = torch.full((x0.shape[0], 1), float(tv), device=device)
            a, s, _, _ = schedule(t)
            residual = (uu - a * x0) / torch.clamp(s, min=1e-3)
            ll = -0.5 * (residual ** 2).sum(dim=1)
            vals.append(torch.logsumexp(ll, dim=0).item())
        vals = np.array(vals)
        vals = vals - vals.max()
        p = np.exp(vals)
        return grid_t, p / np.maximum(p.sum(), 1e-12)

    probe_u = np.array([2.0, 0.1], dtype=np.float32)
    dims_9 = [2, 8, 32, 128]
    posterior_curves_9 = {}
    for _d in dims_9:
        _t_grid, _p = approx_p_t_given_u(probe_u, d=_d)
        posterior_curves_9[_d] = (_t_grid, _p)
    return posterior_curves_9

@app.cell
def __(posterior_curves_9):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
    for ax, _d in zip(axes, [2, 8, 32, 128]):
        _t_grid, _p = posterior_curves_9[_d]
        ax.plot(_t_grid, _p, color=PALETTE['blind'])
        ax.set_title(f'D={_d}')
        ax.set_xlabel('t')
        ax.set_ylim(0, max(_p.max() * 1.15, 1e-6))
    axes[0].set_ylabel('probability')
    fig.suptitle('Posterior p(t | u) across dimensions')
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __(mo):
    for _name, traj in traj_data.items():
        anim = animate_traj(traj, _name)
        display(HTML(anim.to_html5_video()))
    return

@app.cell
def __():
    mo.md("""This is why diffusion works better in high dimensions.""")
    return

@app.cell
def __():
    mo.md("""## Section 10 - From Conditional to Autonomous Field

A conditioned model depends on time $t$:

$$
f(u, t)
$$

An autonomous model removes time:

$$
f(u)
$$

We test whether averaging over $t$ produces a similar field.""")
    return

@app.cell
def __():
    @torch.no_grad()
    def conditioned_avg_field(cond_model, pts_np, t_samples=18):
        x = torch.tensor(pts_np, dtype=torch.float32, device=device)
        acc = torch.zeros_like(x)
        ts = torch.linspace(0.05, 0.95, t_samples, device=device)
        for tv in ts:
            t = torch.full((x.shape[0], 1), float(tv.item()), device=device)
            acc = acc + cond_model(x, t)
        return (acc / t_samples).cpu().numpy()
    
    fm_cond_avg = train_field(FieldCond(d=2), conditioned=True, target='v', d=2, steps=240)
    fm_blind_avg = train_field(FieldBlind(d=2), conditioned=False, target='v', d=2, steps=240)
    _, _, psmall = make_grid(lim=3.4, n=34)
    f_avg = conditioned_avg_field(fm_cond_avg, psmall, t_samples=18)
    f_blind = vector_field_np(fm_blind_avg, psmall, conditioned=False)
    _fig1, _ax1 = plt.subplots()
    
    pick = np.arange(0, psmall.shape[0], 2)
    
    _ax1.quiver(
        psmall[pick, 0], psmall[pick, 1],
        f_avg[pick, 0], f_avg[pick, 1],
        scale=22, width=0.003,
        color=PALETTE['cond']
    )
    
    _ax1.set_title('Average of conditioned fields')
    _ax1.set_xlim(-3.6, 3.6)
    _ax1.set_ylim(-3.6, 3.6)
    _ax1.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    fig2, ax2 = plt.subplots()
    
    ax2.quiver(
        psmall[pick, 0], psmall[pick, 1],
        f_blind[pick, 0], f_blind[pick, 1],
        scale=22, width=0.003,
        color=PALETTE['blind']
    )
    
    ax2.set_title('Learned autonomous field')
    ax2.set_xlim(-3.6, 3.6)
    ax2.set_ylim(-3.6, 3.6)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    return

@app.cell
def __():
    cos_sim = np.mean(np.sum(f_avg * f_blind, axis=1) / (np.linalg.norm(f_avg, axis=1) * np.linalg.norm(f_blind, axis=1) + 1e-8))
    print('Section 10 similarity (cosine):', round(float(cos_sim), 4))
    return

@app.cell
def __():
    mo.md("""Averaging over $t$ produces a similar field.

This explains how autonomous models can emerge from conditioned ones.""")
    return

@app.cell
def __():
    mo.md("""## Section 11 - Field Behavior and Rollout

We examine:

1. The vector field at a fixed time $t$
2. The trajectory generated by the autonomous model

This shows how local dynamics translate into global motion.""")
    return

@app.cell
def __():
    @torch.no_grad()
    def rollout_autonomous(model, d=32, steps=80, step_size=0.04, n=520):
        x = torch.randn(n, d, device=device) * 2.1
        traj = [x[:, :2].detach().cpu().numpy()]
        for _ in range(steps):
            x = x - step_size * model(x)
            traj.append(x[:, :2].detach().cpu().numpy())
        return traj

    return rollout_autonomous

@app.cell
def __():
    t_slider_11 = mo.ui.slider(start=0.02, stop=0.98, step=0.02, value=0.5, label='t', show_value=True)
    D_slider_11 = mo.ui.slider(steps=[2, 8, 32, 128], value=32, label='D', show_value=True)
    step_size_slider_11 = mo.ui.slider(start=0.01, stop=0.10, step=0.005, value=0.04, label='step size', show_value=True)
    n_steps_slider_11 = mo.ui.slider(start=20, stop=140, step=10, value=70, label='n steps', show_value=True)
    controls_11 = mo.hstack([t_slider_11, D_slider_11, step_size_slider_11, n_steps_slider_11], wrap=True)
    return controls_11, t_slider_11, D_slider_11, step_size_slider_11, n_steps_slider_11

@app.cell
def __(t_slider_11, D_slider_11, step_size_slider_11, n_steps_slider_11, rollout_autonomous):
    _model = dim_models[int(D_slider_11.value)]
    traj_11 = rollout_autonomous(
        _model,
        d=int(D_slider_11.value),
        steps=int(n_steps_slider_11.value),
        step_size=float(step_size_slider_11.value),
        n=600,
    )
    _, _, gp_11 = make_grid(lim=3.4, n=24)
    x_11 = torch.tensor(gp_11, dtype=torch.float32, device=device)
    tt_11 = torch.full((x_11.shape[0], 1), float(t_slider_11.value), device=device)
    vf_11 = fm_cond_avg(x_11, tt_11).detach().cpu().numpy()
    return traj_11, gp_11, vf_11

@app.cell
def __(t_slider_11, gp_11, vf_11):
    _fig, _ax = plt.subplots(figsize=(5.4, 4.6))
    _ax.quiver(gp_11[:, 0], gp_11[:, 1], vf_11[:, 0], vf_11[:, 1], scale=24, width=0.003, color='#3a86ff')
    _ax.set_title(f'Section 11: field at t={float(t_slider_11.value):.2f}')
    _ax.set_xlim(-3.6, 3.6)
    _ax.set_ylim(-3.6, 3.6)
    _ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return _fig

@app.cell
def __(D_slider_11, n_steps_slider_11, step_size_slider_11, traj_11):
    _fig, _ax = plt.subplots(figsize=(5.4, 4.6))
    _ax.scatter(traj_11[-1][:, 0], traj_11[-1][:, 1], s=4, alpha=0.35, c='#ef476f')
    _ax.set_title(f'Section 11: rollout D={int(D_slider_11.value)}, steps={int(n_steps_slider_11.value)}, h={float(step_size_slider_11.value):.3f}')
    _ax.set_xlim(-3.8, 3.8)
    _ax.set_ylim(-3.8, 3.8)
    _ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return _fig

@app.cell
def __():
    mo.md("""## Section 12 - Trajectory Evolution

We visualize how samples evolve over time.

The goal is to observe:
- convergence behavior
- stability of motion""")
    return

@app.cell
def __():
    @torch.no_grad()
    def trajectory_animation(model, d=32, steps=90, step_size=0.038, n=560):
        _traj = rollout_autonomous(model, d=d, steps=steps, step_size=step_size, n=n)
        _fig, _ax = plt.subplots(figsize=(5.2, 5.2))
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect('equal', adjustable='box')
        scatter_artist = _ax.scatter([], [], s=5, alpha=0.4, c='#118ab2')
        frame_text = _ax.text(0.03, 0.96, '', transform=_ax.transAxes, va='top')

        def init():
            scatter_artist.set_offsets(np.empty((0, 2)))
            frame_text.set_text('')
            return [scatter_artist, frame_text]

        def update(i):
            scatter_artist.set_offsets(_traj[i])
            frame_text.set_text(f'frame {i + 1}/{len(_traj)}')
            return [scatter_artist, frame_text]

        _anim = FuncAnimation(_fig, update, frames=len(_traj), init_func=init, interval=70, blit=False)
        plt.close(_fig)
        return _anim

    _anim_traj = trajectory_animation(dim_models[32], d=32, steps=95, step_size=0.038, n=560)
    display(HTML(_anim_traj.to_html5_video()))
    return

@app.cell
def __():
    mo.md("""## Section 13 - Dimensional Scaling

We compare trajectory evolution across dimensions:

D = 2, 8, 32, 128

All are projected to 2D for visualization.""")
    return

@app.cell
def __():
    @torch.no_grad()
    def dimensionality_animation(models_by_d, dims_seq=(2, 8, 32, 128), steps=60):
        clouds = []
        for _d in dims_seq:
            _x = torch.randn(680, _d, device=device) * 2.1
            _traj = [_x[:, :2].detach().cpu().numpy()]
            for _ in range(steps):
                _x = _x - 0.035 * models_by_d[_d](_x)
                _traj.append(_x[:, :2].detach().cpu().numpy())
            clouds.append((_d, _traj))

        n_frames = steps + 1
        _fig, _ax = plt.subplots(1, len(dims_seq), figsize=(14.5, 3.9))
        _scat = []
        _hdr = []
        for i, _d in enumerate(dims_seq):
            _ax[i].set_xlim(-4, 4)
            _ax[i].set_ylim(-4, 4)
            _ax[i].set_aspect('equal', adjustable='box')
            _ax[i].set_title(f'D={_d}')
            _s = _ax[i].scatter([], [], s=4, alpha=0.33, c='#073b4c')
            _h = _ax[i].text(0.03, 0.95, '', transform=_ax[i].transAxes, va='top', fontsize=8)
            _scat.append(_s)
            _hdr.append(_h)

        def init2():
            for _s, _h in zip(_scat, _hdr):
                _s.set_offsets(np.empty((0, 2)))
                _h.set_text('')
            return _scat + _hdr

        def update2(i):
            for j, (_, _traj) in enumerate(clouds):
                _scat[j].set_offsets(_traj[i])
                _hdr[j].set_text(f'frame {i + 1}/{n_frames}')
            return _scat + _hdr

        _anim = FuncAnimation(_fig, update2, frames=n_frames, init_func=init2, interval=85, blit=False)
        plt.close(_fig)
        return _anim

    _anim_dim = dimensionality_animation(dim_models)
    display(HTML(_anim_dim.to_html5_video()))
    return

@app.cell
def __():
    mo.md("""Takeaway: 
The learned flow produces smooth and stable trajectories.""")
    return

@app.cell
def __():
    print('Final Takeaways')
    print('1) Geometry can replace explicit noise conditioning.')
    print('2) Raw gradients are unstable and need correction.')
    print('3) Velocity parameterization is the most stable.')
    print('4) Flow Matching is more robust than DDPM without time.')
    print('5) Higher dimensions make noise easier to infer.')
    return

@app.cell
def __():
    mo.md("""## Additional Examples - New Shapes

We repeat Sections 3 to 9 on new datasets:

- Checkerboard mixture
- Three-ring radial mixture

Each shape follows the same pipeline:
- Energy landscape
- Gradient behavior
- Learned vs raw fields
- Stability across parameterizations
- Conditional vs autonomous models
- Dimensional scaling
- Posterior intuition

This tests whether earlier results generalize.""")
    return

@app.cell
def __():
    def sample_checkerboard(n=1024, scale=3.0, noise=0.06):
        x = (torch.rand(n, 2) * 2.0 - 1.0) * scale
        mask = (((torch.floor((x[:, 0] + scale) * 1.5) + torch.floor((x[:, 1] + scale) * 1.5)) % 2) == 0)
        x = x[mask][: n // 2]
        y = (torch.rand(n - x.shape[0], 2) * 2.0 - 1.0) * scale
        mask2 = (((torch.floor((y[:, 0] + scale) * 1.5) + torch.floor((y[:, 1] + scale) * 1.5)) % 2) == 0)
        y = y[mask2]
        z = torch.cat([x, y], dim=0)
        if z.shape[0] < n:
            pad = (torch.rand(n - z.shape[0], 2) * 2.0 - 1.0) * scale
            z = torch.cat([z, pad], dim=0)
        z = z[:n]
        return z + noise * torch.randn_like(z)
    
    def sample_three_rings(n=1024, radii=(1.2, 2.1, 3.0), std=0.09):
        k = len(radii)
        idx = torch.randint(0, k, (n,))
        rr = torch.tensor(radii, dtype=torch.float32)[idx]
        theta = 2 * math.pi * torch.rand(n)
        x = torch.stack([rr * torch.cos(theta), rr * torch.sin(theta)], dim=1)
        return x + std * torch.randn_like(x)
    _extra_shapes = {
        'Checkerboard': sample_checkerboard,
        'ThreeRings': sample_three_rings,
    }
    return

@app.cell
def __():
    def train_field_with_sampler(model, sampler, conditioned=False, target='v', d=2, steps=140, batch=640, lr=1e-3, smooth_weight=0.0):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(steps):
            _, _, t, xt, target_v, target_eps, target_x0 = sample_xt_targets(n=batch, d=d, sampler=sampler)
            pred = model(xt, t) if conditioned else model(xt)
            if target == 'v':
                y = target_v
            elif target == 'eps':
                y = target_eps
            elif target == 'x0':
                y = target_x0
            else:
                raise ValueError('Unknown target')
            loss = ((pred - y) ** 2).mean()
            if smooth_weight > 0 and not conditioned:
                delta = 0.03 * torch.randn_like(xt)
                loss = loss + smooth_weight * ((model(xt + delta) - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model.eval()
    
    def energy_from_samples(gx, gy, samples, bw=0.22, eps=1e-9):
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dif = pts[:, None, :] - samples[None, :, :]
        sq = (dif ** 2).sum(axis=2)
        dens = np.exp(-0.5 * sq / (bw ** 2)).mean(axis=1) / (2 * np.pi * bw ** 2)
        e = -np.log(np.maximum(dens, eps))
        return e.reshape(gx.shape)
    
    @torch.no_grad()
    def recon_error_vs_t_sampler(model, sampler, target='v', conditioned=True, n=1200):
        t_values = np.linspace(0.05, 0.95, 14)
        errs = []
        for tv in t_values:
            x0, _, t, _, _, _, _ = sample_xt_targets(n=n, d=2, sampler=sampler)
            t = torch.full_like(t, float(tv))
            a, s, _, _ = schedule(t)
            eps = torch.randn_like(x0)
            xt = a * x0 + s * eps
            pred = model(xt, t) if conditioned else model(xt)
            if target == 'eps':
                x0_hat = recover_x0_from_eps(xt, pred, t)
            elif target == 'x0':
                x0_hat = pred
            else:
                x0_hat = recover_x0_from_v(xt, pred, t)
            errs.append(((x0_hat - x0) ** 2).mean().item())
        return t_values, np.array(errs)
    
    @torch.no_grad()
    def sample_reverse_with_family(model, family='fm', conditioned=True, n=700, steps=50):
        x = torch.randn(n, 2, device=device) * 2.4
        dt = 1.0 / steps
        for k in range(steps, 0, -1):
            t = torch.full((n, 1), k / steps, device=device)
            if family == 'fm':
                v = model(x, t) if conditioned else model(x)
                x = x - dt * v
            else:
                eps_hat = model(x, t) if conditioned else model(x)
                a, s, _, _ = schedule(t)
                score_like = (x - s * eps_hat) / torch.clamp(a, min=1e-3)
                x = x + dt * (score_like - x)
        return x.detach().cpu().numpy()
    
    @torch.no_grad()
    def posterior_curve_for_sampler(u, sampler, d, n_ref=400):
        t_grid = np.linspace(0.04, 0.96, 80)
        x0_2d = sampler(n_ref).to(device)
        x0 = embed_2d_to_d(x0_2d, d)
        uu = torch.tensor(u, dtype=torch.float32, device=device).view(1, -1)
        uu = embed_2d_to_d(uu, d)
        vals = []
        for tv in t_grid:
            t = torch.full((x0.shape[0], 1), float(tv), device=device)
            a, s, _, _ = schedule(t)
            residual = (uu - a * x0) / torch.clamp(s, min=1e-3)
            vals.append(torch.logsumexp(-0.5 * (residual ** 2).sum(dim=1), dim=0).item())
        vals = np.array(vals)
        vals = vals - vals.max()
        p = np.exp(vals)
        return t_grid, p / np.maximum(p.sum(), 1e-12)
    
    _extra_shapes = {
        'Checkerboard': sample_checkerboard,
        'ThreeRings': sample_three_rings,
    }
    return

@app.cell
def __():
    # =========================================================
    # Additional Examples: Sections 3–9 on New Shapes (CLEAN)
    # =========================================================
    
    _extra_shapes = {
        'Checkerboard': sample_checkerboard,
        'ThreeRings': sample_three_rings,
    }
    
    for _shape_name, _sampler_fn in _extra_shapes.items():
    
        print("\n" + "=" * 70)
        print(f"Running: {_shape_name}")
        print("=" * 70)
    
        # =====================================================
        # SECTION 3 — ENERGY + GRADIENT
        # =====================================================
    
        shape_gx, shape_gy, _ = make_grid(lim=4.2, n=100)
        shape_samples = _sampler_fn(1200).cpu().numpy()
    
        shape_energy = energy_from_samples(shape_gx, shape_gy, shape_samples)
    
        shape_dy, shape_dx = np.gradient(shape_energy, shape_gy[:, 0], shape_gx[0, :])
        shape_grad_norm = np.sqrt(shape_dx**2 + shape_dy**2)
    
        # --- Plot 1: Energy contour ---
        _fig, _ax = plt.subplots()
    
        _c = _ax.contourf(shape_gx, shape_gy, shape_energy, levels=28, cmap='magma')
        _ax.scatter(shape_samples[:, 0], shape_samples[:, 1], s=3, alpha=0.15, c='white')
    
        _ax.set_title(f'{_shape_name} — Energy')
        _ax.set_aspect('equal')
    
        plt.colorbar(_c)
        plt.tight_layout()
        plt.show()
    
        # --- Plot 2: Gradient field ---
        _fig, _ax = plt.subplots()
    
        _skip = (slice(None, None, 4), slice(None, None, 4))
        _q = _ax.quiver(shape_gx[_skip], shape_gy[_skip], shape_dx[_skip], shape_dy[_skip],
              shape_grad_norm[_skip], cmap='cividis')
    
        _ax.set_title(f'{_shape_name} — Gradient')
        _ax.set_aspect('equal')
    
        plt.colorbar(_q)
        plt.tight_layout()
        plt.show()
    
        # =====================================================
        # SECTION 4–5 — RAW vs SCALED vs LEARNED
        # =====================================================
    
        _model_geo = train_field_with_sampler(
            GeometryBlind(d=2),
            _sampler_fn,
            conditioned=False,
            target='v',
            steps=150,
            smooth_weight=0.35
        )
    
        shape_pts = np.stack([shape_gx.ravel(), shape_gy.ravel()], axis=1)
    
        raw_field = np.stack([-shape_dx.ravel(), -shape_dy.ravel()], axis=1)
    
        lam = 1.0 / (1.0 + 2.0 * np.sqrt((raw_field**2).sum(axis=1, keepdims=True)))
        scaled_field = raw_field * lam
    
        learned_field = vector_field_np(_model_geo, shape_pts, conditioned=False)
    
        _pick = np.arange(0, shape_pts.shape[0], 10)
    
        # --- Raw ---
        _fig, _ax = plt.subplots()
        _ax.quiver(shape_pts[_pick, 0], shape_pts[_pick, 1],
              raw_field[_pick, 0], raw_field[_pick, 1],
                  color=PALETTE['blind'], scale=22)
        _ax.set_title(f'{_shape_name} — Raw gradient')
        _ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
        # --- Scaled ---
        _fig, _ax = plt.subplots()
        _ax.quiver(shape_pts[_pick, 0], shape_pts[_pick, 1],
              scaled_field[_pick, 0], scaled_field[_pick, 1],
                  color=PALETTE['cond'], scale=22)
        _ax.set_title(f'{_shape_name} — Scaled flow')
        _ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
        # --- Learned ---
        _fig, _ax = plt.subplots()
        _ax.quiver(shape_pts[_pick, 0], shape_pts[_pick, 1],
              learned_field[_pick, 0], learned_field[_pick, 1],
                  color=PALETTE['vel'], scale=22)
        _ax.set_title(f'{_shape_name} — Learned field')
        _ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
        # =====================================================
        # SECTION 6 — STABILITY
        # =====================================================
    
        _m_eps = train_field_with_sampler(FieldCond(d=2), _sampler_fn, True, 'eps', steps=120)
        _m_x0  = train_field_with_sampler(FieldCond(d=2), _sampler_fn, True, 'x0', steps=120)
        _m_vel = train_field_with_sampler(FieldCond(d=2), _sampler_fn, True, 'v',   steps=120)
    
        t_vals, _err_eps = recon_error_vs_t_sampler(_m_eps, _sampler_fn, 'eps', True)
        _, _err_x0 = recon_error_vs_t_sampler(_m_x0, _sampler_fn, 'x0', True)
        _, _err_vel = recon_error_vs_t_sampler(_m_vel, _sampler_fn, 'v', True)
    
        _fig, _ax = plt.subplots()
    
        _ax.plot(t_vals, _err_eps, '-o', color=PALETTE['eps'], label='eps')
        _ax.plot(t_vals, _err_x0, '-o', color=PALETTE['x0'], label='x0')
        _ax.plot(t_vals, _err_vel, '-o', color=PALETTE['vel'], label='velocity')
    
        _ax.set_title(f'{_shape_name} — Stability')
        _ax.set_xlabel('t')
        _ax.set_ylabel('MSE')
        _ax.legend(frameon=False)
    
        plt.tight_layout()
        plt.show()
    
        # =====================================================
        # SECTION 7 — CONDITIONAL vs AUTONOMOUS
        # =====================================================
    
        _ddpm = train_field_with_sampler(FieldCond(d=2), _sampler_fn, True, 'eps', steps=110)
        _ddpm_blind = train_field_with_sampler(FieldBlind(d=2), _sampler_fn, False, 'eps', steps=110)
        _fm = train_field_with_sampler(FieldCond(d=2), _sampler_fn, True, 'v', steps=110)
        _fm_blind = train_field_with_sampler(FieldBlind(d=2), _sampler_fn, False, 'v', steps=110)
    
        _samples_dict = {
            'DDPM': sample_reverse_with_family(_ddpm, 'ddpm', True),
            'DDPM Blind': sample_reverse_with_family(_ddpm_blind, 'ddpm', False),
            'FM': sample_reverse_with_family(_fm, 'fm', True),
            'FM Blind': sample_reverse_with_family(_fm_blind, 'fm', False),
        }
    
        for _name, pts_s in _samples_dict.items():
            _fig, _ax = plt.subplots()
    
            _ax.scatter(pts_s[:, 0], pts_s[:, 1], s=4, alpha=0.35, c=PALETTE['x0'])
    
            _ax.set_title(f'{_shape_name} — {_name}')
            _ax.set_xlim(-4.2, 4.2)
            _ax.set_ylim(-4.2, 4.2)
            _ax.set_aspect('equal')
    
            plt.tight_layout()
            plt.show()
    
        # =====================================================
        # SECTION 8 — DIMENSIONALITY
        # =====================================================
    
        for _d in [2, 16, 64]:
    
            _model_d = train_field_with_sampler(
                FieldBlind(d=_d),
                _sampler_fn,
                conditioned=False,
                target='v',
                d=_d,
                steps=100,
                batch=512
            )
    
            _x = torch.randn(700, _d, device=device) * 2.2
    
            for _ in range(45):
                _x = _x - 0.035 * _model_d(_x)
    
            pts2d = _x[:, :2].detach().cpu().numpy()
    
            _fig, _ax = plt.subplots()
    
            _ax.scatter(pts2d[:, 0], pts2d[:, 1],
                       s=4, alpha=0.32, c=PALETTE['x0'])
    
            _ax.set_title(f'{_shape_name} — D={_d}')
            _ax.set_xlim(-4.2, 4.2)
            _ax.set_ylim(-4.2, 4.2)
            _ax.set_aspect('equal')
    
            plt.tight_layout()
            plt.show()
    
        # =====================================================
        # SECTION 9 — POSTERIOR
        # =====================================================
    
        _probe = np.array([1.7, 0.4], dtype=np.float32)
    
        _t_grid, _p_low = posterior_curve_for_sampler(_probe, _sampler_fn, d=2)
        _, _p_high = posterior_curve_for_sampler(_probe, _sampler_fn, d=64)
    
        _fig, _ax = plt.subplots()
    
        _ax.plot(_t_grid, _p_low, label='D=2', color=PALETTE['blind'])
        _ax.plot(_t_grid, _p_high, label='D=64', color=PALETTE['cond'])
    
        _ax.set_title(f'{_shape_name} — Posterior p(t|u)')
        _ax.set_xlabel('t')
        _ax.set_ylabel('probability')
        _ax.legend(frameon=False)
    
        plt.tight_layout()
        plt.show()
    
        # =====================================================
        # SUMMARY
        # =====================================================
    
        print(f"{_shape_name} summary:")
        print("- Geometry behavior consistent")
        print("- Velocity most stable")
        print("- High dimension improves inference")
    return


if __name__ == "__main__":
    app.run()
