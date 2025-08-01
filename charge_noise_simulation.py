#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charge Noise Simulation Script

This script simulates charge noise and analyzes its spectral characteristics, 
checking if 1/f noise can be obtained.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import pandas as pd
import colorednoise as cn

# Set random seed for reproducible results
# np.random.seed(42)

def generate_charge_noise(n_samples, dt, method='white'):
    """Generate charge noise time series"""
    
    # Parameter settings
    total_time = n_samples * dt
    
    # Generate time axis
    t = np.linspace(0, total_time, n_samples)
    
    print(f"Time series length: {n_samples} samples")
    print(f"Sampling interval: {dt*1000:.2f} milliseconds")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Sampling frequency: {1/dt:.0f} Hz")
    
    # Generate base signal (can be DC bias)
    base_signal = 3e-10  # Base bias
    fluctuation_amplitude = 2e-10
    
    # Generate different types of noise
    
    # 1. White noise (Gaussian noise)
    white_noise = np.random.normal(0, fluctuation_amplitude, n_samples)
    
    # 2. 1/f noise (pink noise)
    # Generate 1/f noise by filtering white noise
    frequencies = fftfreq(n_samples, dt)
    positive_freq_mask = frequencies > 0
    
    if method == 'white':
        # Generate 1/f noise
        white_noise_fft = fft(white_noise)
        pink_noise_fft = white_noise_fft.copy()
        pink_noise_fft[positive_freq_mask] /= np.sqrt(frequencies[positive_freq_mask])  # A(f) = 1/sqrt(f)
        pink_noise_fft[~positive_freq_mask] = 0  # Handle negative frequencies
        pink_noise = np.real(np.fft.ifft(pink_noise_fft))
    elif method == 'pink':
        # generate directly 
        pink_noise = fluctuation_amplitude * cn.powerlaw_psd_gaussian(1, n_samples)
    else:
        raise ValueError(f"Invalid method: {method}")

    # 3. Random walk noise (Brownian noise)
    brown_noise = np.cumsum(white_noise) * 0.01
    
    # 4. Periodic noise (simulate interference)
    periodic_noise = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 120 * t)
    
    # Combine all noise components
    # total_noise = 0.1 * white_noise + 0.5 * pink_noise + 0.3 * brown_noise 
    total_noise = pink_noise
    
    # Final signal
    signal_with_noise = base_signal + total_noise

    return t, signal_with_noise, white_noise, pink_noise, base_signal

def plot_time_series(t, signal_with_noise, white_noise, pink_noise):
    """Plot time series graphs"""
    
    # Plot time series
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Complete time series
    axes[0].set_xlim(min(t), max(t))
    axes[0].plot(t, signal_with_noise, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Signal Amplitude')
    axes[0].set_title('Complete Time Series')
    axes[0].grid(True, alpha=0.3)
    
    # Signal histogram
    axes[1].hist(signal_with_noise, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Signal Amplitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Signal Amplitude Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')

def analyze_spectrum_welch(signal_with_noise, dt):
    """Analyze spectral characteristics using Welch method"""

    # welch method - returns only non-negative frequencies
    frequencies, psd = welch(signal_with_noise, fs=1/dt, nperseg=1024)
    
    return frequencies, psd

def analyze_spectrum_fft(signal_with_noise, dt):
    """Analyze spectral characteristics"""
    
    # Calculate Fourier transform
    fft_signal = fft(signal_with_noise)
    frequencies = fftfreq(len(signal_with_noise), dt)
    
    # Calculate power spectral density
    psd = np.abs(fft_signal)**2

    # Only consider positive frequency part
    positive_freq_mask = frequencies > 0
    freq_positive = frequencies[positive_freq_mask]
    psd_positive = psd[positive_freq_mask]
    
    return freq_positive, psd_positive

def plot_spectrum(freq_positive, psd_positive, method):
    """Plot spectrum graphs with 1/f reference"""
    
    # Plot spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log-log scale spectrum with 1/f reference
    axes[0].loglog(freq_positive, psd_positive, 'r-', linewidth=0.8, label='Measured Data')

    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('Power Spectral Density')
    axes[0].set_title(f'Power Spectral Density ({method}) with 1/f Reference')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectral density distribution
    axes[1].hist(np.log10(psd_positive), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('log10(Power Spectral Density)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Power Spectral Density Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'spectrum_analysis_{method}.png', dpi=300, bbox_inches='tight')

n_samples = 20000
dt = 50e-3  # 50 milliseconds

# Generate charge noise
t, signal_with_noise, white_noise, pink_noise, base_signal = generate_charge_noise(n_samples, dt, method='pink')
    
# Plot time series
plot_time_series(t, signal_with_noise, white_noise, pink_noise)
    
# Spectral analysis (Welch)
freq_welch, psd_welch = analyze_spectrum_welch(signal_with_noise, dt)
plot_spectrum(freq_welch, psd_welch, "Welch")

# Spectral analysis (FFT)
freq_positive, psd_positive = analyze_spectrum_fft(signal_with_noise, dt)
plot_spectrum(freq_positive, psd_positive, "FFT")