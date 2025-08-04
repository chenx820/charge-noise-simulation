"""
Reference: https://ieeexplore.ieee.org/document/9785516
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

def generate_lorentzian_psd(
    W, L, tox, Nt, delta_E,
    num_devices=1,
    f_min=1.0, f_max=1e6, points_per_decade=10):
    """
    Generate Lorentzian noise power spectral density (PSD) for one or multiple devices
    based on the paper methodology.
    
    Args:
        W (float): Device width (m)
        L (float): Device length (m)
        tox (float): Oxide layer thickness (m)
        Nt (float): Oxide trap density (1 / (m^3 * eV))
        delta_E (float): Total energy bandgap (eV)
        num_devices (int): Number of devices (chips) to simulate
        f_min (float): Minimum frequency range (Hz)
        f_max (float): Maximum frequency range (Hz)
        points_per_decade (int): Number of frequency points per decade

    Returns:
        tuple: (frequency array, PSD array list)
               One PSD array for each device
    """
    # Physical constants
    q = 1.602e-19  # Elementary charge (C)
    Cox_per_area = 3.9 * 8.854e-12 / tox # Oxide capacitance per unit area (F/m^2)

    # --- Step 1: Determine trap count ---
    # Calculate average trap number <N_T>, reference paper
    N_T_avg = Nt * W * L * tox * delta_E
    
    # Sample actual trap count N_T for each device from Poisson distribution
    N_T_per_device = np.random.poisson(N_T_avg, num_devices)

    # Generate logarithmically distributed frequency points
    freq = np.logspace(np.log10(f_min), np.log10(f_max), 
                       int(np.log10(f_max/f_min) * points_per_decade))
    
    all_psds = []

    print(f"Average trap count <N_T>: {N_T_avg:.2f}")
    
    for i, N_T in enumerate(N_T_per_device):
        print(f"Simulating device {i+1}: has {N_T} traps...")
        
        if N_T == 0:
            all_psds.append(np.zeros_like(freq))
            continue

        # --- Step 2: Generate characteristics for each trap ---
        trap_params = []
        for k in range(N_T):
            # The paper uses some simplified methods (Eq. 7) to model time and charge
            # Here we directly randomize the core parameters tau_k and Delta_Vt_k
            
            # Randomly generate trap depth xt (uniform distribution between 0 and tox)
            xt_k = np.random.uniform(0, tox)
            
            # Calculate Delta_Vt_k according to Eq. (5b)
            delta_Vt_k = q * (1 - xt_k / tox) / (W * L * Cox_per_area)
            
            # Randomly generate characteristic time constant tau_k
            # The paper mentions that tau_0 is exponentially distributed
            # Here we simplify to uniform distribution in log space to cover multiple decades
            log_tau_k = np.random.uniform(-9, -3) # 1ns to 1ms
            tau_k = 10**log_tau_k
            
            # Calculate A_k, simplified assumption that capture and emission times are roughly equal
            # Real calculation depends on Eq. (5c) and Eq. (7)
            A_k = 0.25 # When tau_c ~ tau_e, A ~ 0.25
            
            trap_params.append({'delta_Vt': delta_Vt_k, 'tau': tau_k, 'A': A_k})

        # --- Step 3: Superimpose noise spectra from all traps ---
        total_psd = np.zeros_like(freq)
        
        # Use Eq. (5a)
        for params in trap_params:
            delta_Vt = params['delta_Vt']
            tau = params['tau']
            A = params['A']
            
            numerator = 4 * (delta_Vt**2) * A * tau
            denominator = 1 + (2 * np.pi * freq * tau)**2
            
            single_trap_psd = numerator / denominator
            total_psd += single_trap_psd
            
        all_psds.append(total_psd)
        
    return freq, all_psds


# --- Define device and process parameters ---
# Using dimensions of our device SemiQon 36
W_eff = 75e-9  # Effective width (m)
L_eff = 360e-9   # Length (m)
tox_eq = 17e-9   # Equivalent oxide thickness (m)

# Assumed process parameters
Nt_val = 5e23  # (1/(m^3*eV)) - This is a typical value
delta_E_val = 0.1 # (eV) - Effective energy range near Fermi level

# --- Run simulation ---
# Mimic Figure 1 in the paper, simulate noise variation across multiple devices (dies)
num_sim_devices = 200 
frequencies, psd_list = generate_lorentzian_psd(
        W=W_eff, L=L_eff, tox=tox_eq, Nt=Nt_val, delta_E=delta_E_val,
        num_devices=num_sim_devices
)

# --- Plotting ---
# Plot decorators
plt.rc("legend", fontsize=22, framealpha=0.9)
plt.rc("xtick", labelsize=24, color="#2C3E50")
plt.rc("ytick", labelsize=24, color="#2C3E50")

fig, ax = plt.subplots(figsize=(12, 7))
    
# Plot PSD spectrum for each device
for psd in psd_list:
    ax.loglog(frequencies, psd, alpha=0.5)

# Calculate and plot average spectrum
average_psd = np.mean(psd_list, axis=0)
ax.loglog(frequencies, average_psd, color='black', linewidth=2.5, label=f'Average of {num_sim_devices} devices')

ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Power Spectral Density S$_{Vg}$ (V²/Hz)', fontsize=24)
ax.grid(True, which="both", ls="--")
ax.legend(fontsize=24)

plt.tight_layout()
dir_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(dir_path, 'figures', 'mosfet_lorentzian_noise_simulation.png'), dpi=300)