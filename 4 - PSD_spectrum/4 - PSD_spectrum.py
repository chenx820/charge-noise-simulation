import numpy as np
import matplotlib.pyplot as plt
import os


def single_psd(f, tau, delta_mu = 1):
    return delta_mu * tau / (4 + (2 * np.pi * f * tau) ** 2)

# Simulation parameters
n_realizations = 50  # Number of realizations
n_tau = 20  # Number of tau values per realization

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

f_list = np.logspace(-2, 1, 20)

# Store PSD results from all realizations
all_psd_uniform = []
all_psd_loguniform = []
all_psd_lognormal = []

# Multiple realizations
for realization in range(n_realizations):
    # Generate tau with different distributions
    tau_uniform = np.random.uniform(0.1, 10, n_tau)
    tau_loguniform = np.exp(np.random.uniform(np.log(0.1), np.log(10), n_tau))
    tau_lognormal = np.random.lognormal(0, 1, n_tau)
    
    # Calculate total PSD for each distribution
    total_psd_uniform = np.zeros_like(f_list)
    total_psd_loguniform = np.zeros_like(f_list)
    total_psd_lognormal = np.zeros_like(f_list)
    
    # Uniform distribution
    for tau in tau_uniform:
        psd = single_psd(f_list, tau)
        if realization < 5:  # Only show first 5 realizations as individual curves
            axes[0].loglog(f_list, psd, alpha=0.3, linewidth=0.5, color='blue')
        total_psd_uniform += psd
    
    # Log-uniform distribution
    for tau in tau_loguniform:
        psd = single_psd(f_list, tau)
        if realization < 5:  # Only show first 5 realizations as individual curves
            axes[1].loglog(f_list, psd, alpha=0.3, linewidth=0.5, color='green')
        total_psd_loguniform += psd
    
    # Log-normal distribution
    for tau in tau_lognormal:
        psd = single_psd(f_list, tau)
        if realization < 5:  # Only show first 5 realizations as individual curves
            axes[2].loglog(f_list, psd, alpha=0.3, linewidth=0.5, color='red')
        total_psd_lognormal += psd
    
    # Store total PSD from each realization
    all_psd_uniform.append(total_psd_uniform)
    all_psd_loguniform.append(total_psd_loguniform)
    all_psd_lognormal.append(total_psd_lognormal)

# Calculate averaged PSD
avg_psd_uniform = np.mean(all_psd_uniform, axis=0)
avg_psd_loguniform = np.mean(all_psd_loguniform, axis=0)
avg_psd_lognormal = np.mean(all_psd_lognormal, axis=0)

# Plot averaged PSD curves
axes[0].loglog(f_list, avg_psd_uniform, color='black', linewidth=2, label='Averaged PSD')
axes[0].set_title('Uniform Distribution')
axes[0].legend()

axes[1].loglog(f_list, avg_psd_loguniform, color='black', linewidth=2, label='Averaged PSD')
axes[1].set_title('Log-uniform Distribution')
axes[1].legend()

axes[2].loglog(f_list, avg_psd_lognormal, color='black', linewidth=2, label='Averaged PSD')
axes[2].set_title('Log-normal Distribution')
axes[2].legend()

# Add 1/f reference lines (optional)
# Calculate appropriate scaling factors
scale_factor_uniform = avg_psd_uniform[len(f_list)//2] * f_list[len(f_list)//2]
scale_factor_loguniform = avg_psd_loguniform[len(f_list)//2] * f_list[len(f_list)//2]
scale_factor_lognormal = avg_psd_lognormal[len(f_list)//2] * f_list[len(f_list)//2]

axes[0].loglog(f_list, scale_factor_uniform / f_list, '--', color='orange', linewidth=1.5, label='1/f reference')
axes[1].loglog(f_list, scale_factor_loguniform / f_list, '--', color='orange', linewidth=1.5, label='1/f reference')
axes[2].loglog(f_list, scale_factor_lognormal / f_list, '--', color='orange', linewidth=1.5, label='1/f reference')

axes[0].legend()
axes[1].legend()
axes[2].legend()

plt.tight_layout()
dir_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(dir_path, 'figures/PSD_spectrum.png'), dpi=300)












