#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pink Noise Analysis Script

This script generates pink noise with different power law exponents (alpha)
and analyzes the spectral characteristics, checking noise levels at specific frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import colorednoise as cn
import os

# Set random seed for reproducible results
# np.random.seed(42)

def generate_pink_noise(n_samples, alpha, normalize=True):
    """
    Generate pink noise with power law 1/f^alpha
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    alpha : float
        Power law exponent (1/f^alpha)
    normalize : bool
        If True, normalize to standard deviation = 1
    
    Returns:
    --------
    pink_noise : array
        Generated pink noise time series
    """
    
    # Generate pink noise using colorednoise library
    pink_noise = cn.powerlaw_psd_gaussian(alpha, n_samples)
    
    if normalize:
        # Normalize to standard deviation = 1
        pink_noise = pink_noise / np.std(pink_noise)
    
    return pink_noise

def analyze_noise_levels(frequencies, psd, target_freqs=[0.1, 1.0, 10.0]):
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

def plot_pink_noise_comparison(t, pink_noises, alphas):
    """Plot comparison of pink noise with different alpha values"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series comparison
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (alpha, noise) in enumerate(zip(alphas, pink_noises)):
        color = colors[i % len(colors)]
        axes[0].plot(t, noise, color=color, alpha=0.7, 
                        linewidth=0.8, label=f'α = {alpha}')
    
    axes[0].set_xlim(min(t), max(t))
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Pink Noise Time Series Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram comparison
    for i, (alpha, noise) in enumerate(zip(alphas, pink_noises)):
        color = colors[i % len(colors)]
        axes[1].hist(noise, bins=50, alpha=0.5, color=color, 
                        label=f'α = {alpha}', density=True)
    
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Amplitude Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dir_path, 'figures', 'pink_noise_comparison.png'), dpi=300, bbox_inches='tight')

def plot_spectrum_analysis(frequencies, psds, alphas, method="Welch"):
    """Plot spectrum analysis for different alpha values"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Log-log spectrum comparison
    for i, (alpha, freq, psd) in enumerate(zip(alphas, frequencies, psds)):
        color = colors[i % len(colors)]
        axes[0].loglog(freq, psd, color=color, alpha=0.7, 
                          linewidth=1.2, label=f'α = {alpha}')
    
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel(r'PSD [a.u.$^2$/Hz]')
    axes[0].set_title(f'Power Spectral Density Comparison ({method})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sqrt frequency spectrum
    for i, (alpha, freq, psd) in enumerate(zip(alphas, frequencies, psds)):
        color = colors[i % len(colors)]
        axes[1].loglog(freq, np.sqrt(psd), color=color, alpha=0.7, 
                          linewidth=1.2, label=f'α = {alpha}')
    

    for x in [0.1, 1, 10]:
        axes[1].axvline(x=x, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)

    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel(r'√PSD [a.u./√Hz]')
    axes[1].set_title('√PSD vs Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dir_path, 'figures', f'pink_noise_spectrum_{method}.png'), dpi=300, bbox_inches='tight')


# Parameters
n_samples = 20000
dt = 50e-3  # 50 milliseconds
alphas = [0.5, 1.0, 1.5, 2.0]  # Different power law exponents
    
# Generate pink noise with different alpha values
pink_noises = []
for alpha in alphas:
    pink_noise = generate_pink_noise(n_samples, alpha, normalize=True)
    pink_noises.append(pink_noise)
    
# Time axis
t = np.linspace(0, n_samples * dt, n_samples)
    
# Plot time series comparison
plot_pink_noise_comparison(t, pink_noises, alphas)
    
# Spectral analysis using Welch method
frequencies_welch = []
psds_welch = []
    
for i, (alpha, noise) in enumerate(zip(alphas, pink_noises)):
    freq, psd = welch(noise, fs=1/dt, nperseg=1024)
    frequencies_welch.append(freq)
    psds_welch.append(psd)
        
    # Analyze noise levels at specific frequencies
    noise_levels = analyze_noise_levels(freq, psd, [0.1, 1.0, 10.0])
    
# Plot spectrum analysis
plot_spectrum_analysis(frequencies_welch, psds_welch, alphas, "Welch")
