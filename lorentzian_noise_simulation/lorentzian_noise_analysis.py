#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorentzian Noise Analysis Script

This script generates Lorentzian noise with different parameters
and analyzes the spectral characteristics, checking noise levels at specific frequencies.
Lorentzian noise is characterized by a Lorentzian power spectral density.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

# Set random seed for reproducible results
np.random.seed(42)

def generate_lorentzian_noise(n_samples, dt, f0, gamma, amplitude=1.0, normalize=True):
    """
    Generate Lorentzian noise with specified parameters
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    dt : float
        Time step
    f0 : float
        Center frequency (Hz)
    gamma : float
        Half-width at half maximum (HWHM) in Hz
    amplitude : float
        Amplitude of the noise
    normalize : bool
        If True, normalize to standard deviation = 1
    
    Returns:
    --------
    lorentzian_noise : array
        Generated Lorentzian noise time series
    """
    
    # Time axis
    t = np.linspace(0, n_samples * dt, n_samples)
    
    # Frequency axis for spectral generation
    freq = np.fft.fftfreq(n_samples, dt)
    freq = freq[:n_samples//2 + 1]  # Positive frequencies only
    
    # Generate Lorentzian power spectral density
    psd = amplitude / (1 + ((freq - f0) / gamma)**2)
    
    # Add small constant to avoid division by zero
    psd += 1e-10
    
    # Generate complex spectrum with random phase
    phase = 2 * np.pi * np.random.rand(len(freq))
    spectrum = np.sqrt(psd) * np.exp(1j * phase)
    
    # Make spectrum conjugate symmetric for real time series
    full_spectrum = np.zeros(n_samples, dtype=complex)
    full_spectrum[:len(spectrum)] = spectrum
    full_spectrum[-len(spectrum)+1:] = np.conj(spectrum[1:])
    
    # Inverse FFT to get time series
    lorentzian_noise = np.real(np.fft.ifft(full_spectrum))
    
    if normalize:
        # Normalize to standard deviation = 1
        lorentzian_noise = lorentzian_noise / np.std(lorentzian_noise)
    
    return lorentzian_noise

def analyze_noise_levels(frequencies, psd, target_freqs=None):
    """
    Analyze noise levels at specific frequencies
    
    Parameters:
    -----------
    frequencies : array
        Frequency array
    psd : array
        Power spectral density
    target_freqs : list
        Target frequencies to analyze
    
    Returns:
    --------
    noise_levels : dict
        Dictionary with noise levels at target frequencies
    """
    
    if target_freqs is None:
        target_freqs = [0.1, 1.0, 10.0]
    
    noise_levels = {}
    
    for freq in target_freqs:
        # Find closest frequency in the data
        idx = np.argmin(np.abs(frequencies - freq))
        actual_freq = frequencies[idx]
        noise_level = psd[idx]
        
        noise_levels[freq] = {
            'actual_freq': actual_freq,
            'noise_level': noise_level,
            'noise_level_sqrt': np.sqrt(noise_level)
        }
    
    return noise_levels

def plot_lorentzian_noise_comparison(t, lorentzian_noises, params_list):
    """Plot comparison of Lorentzian noise with different parameters"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series comparison
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (params, noise) in enumerate(zip(params_list, lorentzian_noises)):
        color = colors[i % len(colors)]
        label = f'f0={params["f0"]}Hz, gamma={params["gamma"]}Hz'
        axes[0].plot(t, noise, color=color, alpha=0.7, 
                        linewidth=0.8, label=label)
    
    axes[0].set_xlim(min(t), max(t))
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Lorentzian Noise Time Series Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram comparison
    for i, (params, noise) in enumerate(zip(params_list, lorentzian_noises)):
        color = colors[i % len(colors)]
        label = f'f0={params["f0"]}Hz, gamma={params["gamma"]}Hz'
        axes[1].hist(noise, bins=50, alpha=0.5, color=color, 
                        label=label, density=True)
    
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Amplitude Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(dir_path, 'figures', 'lorentzian_noise_comparison.png'), dpi=300, bbox_inches='tight')

def plot_spectrum_analysis(frequencies, psds, params_list, method="Welch"):
    """Plot spectrum analysis for different Lorentzian parameters"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Log-log spectrum comparison
    for i, (params, freq, psd) in enumerate(zip(params_list, frequencies, psds)):
        color = colors[i % len(colors)]
        label = fr'$f_0$={params["f0"]}Hz, $\gamma$={params["gamma"]}Hz'
        axes[0].loglog(freq, psd, color=color, alpha=0.7, 
                          linewidth=1.2, label=label)
    
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel(r'PSD [a.u.$^2$/Hz]')
    axes[0].set_title(f'Power Spectral Density Comparison ({method})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Linear scale spectrum
    for i, (params, freq, psd) in enumerate(zip(params_list, frequencies, psds)):
        color = colors[i % len(colors)]
        label = f'f0={params["f0"]}Hz, gamma={params["gamma"]}Hz'
        axes[1].plot(freq, psd, color=color, alpha=0.7, 
                        linewidth=1.2, label=label)

    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel(r'PSD [a.u.$^2$/Hz]')
    axes[1].set_title('Power Spectral Density (Linear Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(dir_path, 'figures', f'lorentzian_noise_spectrum_{method}.png'), dpi=300, bbox_inches='tight')

def plot_theoretical_vs_simulated(freq, psd_sim, f0, gamma, amplitude=1.0):
    """Plot theoretical vs simulated Lorentzian spectrum"""
    
    # Theoretical Lorentzian PSD
    psd_theoretical = amplitude / (1 + (freq / f0)**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log scale comparison
    axes[0].loglog(freq, psd_sim, 'b-', linewidth=1.5, label='Simulated')
    axes[0].loglog(freq, psd_theoretical, 'r--', linewidth=1.5, label='Theoretical')
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel(r'PSD [a.u.$^2$/Hz]')
    axes[0].set_title('Theoretical vs Simulated Lorentzian Spectrum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Linear scale comparison
    axes[1].plot(freq, psd_sim, 'b-', linewidth=1.5, label='Simulated')
    axes[1].plot(freq, psd_theoretical, 'r--', linewidth=1.5, label='Theoretical')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel(r'PSD [a.u.$^2$/Hz]')
    axes[1].set_title('Theoretical vs Simulated Lorentzian Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dir_path, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(dir_path, 'figures', 'theoretical_vs_simulated_lorentzian.png'), dpi=300, bbox_inches='tight')

# Simulation parameters
n_samples = 20000
dt = 50e-3  # 50 milliseconds

# Different Lorentzian noise parameters
params_list = [
    {'f0': 1.0, 'gamma': 0.1, 'amplitude': 1.0},
    {'f0': 2.0, 'gamma': 0.2, 'amplitude': 1.0},
    {'f0': 5.0, 'gamma': 0.5, 'amplitude': 1.0},
    {'f0': 10.0, 'gamma': 1.0, 'amplitude': 1.0}
]

# Generate Lorentzian noise with different parameters
lorentzian_noises = []
for params in params_list:
    lorentzian_noise = generate_lorentzian_noise(
        n_samples, dt, params['f0'], params['gamma'], params['amplitude'], normalize=True
    )
    lorentzian_noises.append(lorentzian_noise)
    
# Create time axis
t = np.linspace(0, n_samples * dt, n_samples)
    
# Plot time series comparison
plot_lorentzian_noise_comparison(t, lorentzian_noises, params_list)
    
# Perform spectral analysis using Welch method
frequencies_welch = []
psds_welch = []
    
for i, (params, noise) in enumerate(zip(params_list, lorentzian_noises)):
    freq, psd = welch(noise, fs=1/dt, nperseg=1024)
    frequencies_welch.append(freq)
    psds_welch.append(psd)
        
    # Analyze noise levels at specific frequencies
    noise_levels = analyze_noise_levels(freq, psd, [0.1, 1.0, 10.0])
    
    print(f"Parameters: f0={params['f0']}Hz, gamma={params['gamma']}Hz")
    for target_freq, levels in noise_levels.items():
        print(f"  Frequency {target_freq}Hz: PSD = {levels['noise_level']:.6f}, ASP = {levels['noise_level_sqrt']:.6f}")
    print()
    
# Plot spectrum analysis
plot_spectrum_analysis(frequencies_welch, psds_welch, params_list, "Welch")

# Plot theoretical vs simulated for one case
freq, psd = welch(lorentzian_noises[0], fs=1/dt, nperseg=1024)
plot_theoretical_vs_simulated(freq, psd, params_list[0]['f0'], params_list[0]['gamma'], params_list[0]['amplitude'])

print("Lorentzian noise simulation completed!")
print("Generated figures:")
print("- lorentzian_noise_comparison.png")
print("- lorentzian_noise_spectrum_Welch.png") 
print("- theoretical_vs_simulated_lorentzian.png") 