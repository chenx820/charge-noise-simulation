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
import seaborn as sns

# Set font for better display
#plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducible results
np.random.seed(42)

def generate_charge_noise():
    """Generate charge noise time series"""
    
    # Parameter settings
    n_samples = 20000
    dt = 50e-3  # 50 milliseconds
    total_time = n_samples * dt
    
    # Generate time axis
    t = np.linspace(0, total_time, n_samples)
    
    print(f"Time series length: {n_samples} samples")
    print(f"Sampling interval: {dt*1000:.2f} milliseconds")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Sampling frequency: {1/dt:.0f} Hz")
    
    # Generate base signal (can be DC bias)
    base_signal = 1.0  # Base bias
    
    # Generate different types of noise
    
    # 1. White noise (Gaussian noise)
    white_noise = np.random.normal(0, 0.1, n_samples)
    print(f"White noise: {white_noise}")
    
    # 2. 1/f noise (pink noise)
    # Generate 1/f noise by filtering white noise
    frequencies = fftfreq(n_samples, dt)
    positive_freq_mask = frequencies > 0
    
    # Generate 1/f noise
    white_noise_fft = fft(white_noise)
    pink_noise_fft = white_noise_fft.copy()
    pink_noise_fft[positive_freq_mask] /= np.sqrt(frequencies[positive_freq_mask])
    pink_noise_fft[~positive_freq_mask] = 0  # Handle negative frequencies
    pink_noise = np.real(np.fft.ifft(pink_noise_fft))
    
    # 3. Random walk noise (Brownian noise)
    brown_noise = np.cumsum(white_noise) * 0.01
    
    # 4. Periodic noise (simulate interference)
    periodic_noise = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 120 * t)
    
    # Combine all noise components
    total_noise = white_noise + 0.5 * pink_noise + 0.3 * brown_noise + periodic_noise
    
    # Final signal
    signal_with_noise = base_signal + total_noise
    
    print(f"\nSignal statistics:")
    print(f"  Mean: {np.mean(signal_with_noise):.4f}")
    print(f"  Standard deviation: {np.std(signal_with_noise):.4f}")
    print(f"  Minimum: {np.min(signal_with_noise):.4f}")
    print(f"  Maximum: {np.max(signal_with_noise):.4f}")
    
    return t, signal_with_noise, white_noise, pink_noise, base_signal

def plot_time_series(t, signal_with_noise, white_noise, pink_noise):
    """Plot time series graphs"""
    
    # Plot time series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Complete time series
    axes[0, 0].plot(t, signal_with_noise, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Signal Amplitude')
    axes[0, 0].set_title('Complete Time Series')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Detailed view of first 1 second
    t_1s = t[t <= 1.0]
    signal_1s = signal_with_noise[:len(t_1s)]
    axes[0, 1].plot(t_1s, signal_1s, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Signal Amplitude')
    axes[0, 1].set_title('First 1 Second Detailed View')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Noise component analysis
    axes[1, 0].plot(t[:1000], white_noise[:1000], 'g-', label='White Noise', alpha=0.7)
    axes[1, 0].plot(t[:1000], pink_noise[:1000], 'm-', label='1/f Noise', alpha=0.7)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Noise Amplitude')
    axes[1, 0].set_title('Noise Component Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Signal histogram
    axes[1, 1].hist(signal_with_noise, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Signal Amplitude')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Signal Amplitude Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_spectrum(signal_with_noise, dt):
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
    
    print(f"\nSpectral analysis:")
    print(f"Frequency range: {freq_positive.min():.2f} Hz to {freq_positive.max():.0f} Hz")
    print(f"Nyquist frequency: {1/(2*dt):.0f} Hz")
    
    return freq_positive, psd_positive

def plot_spectrum(freq_positive, psd_positive):
    """Plot spectrum graphs"""
    
    # Plot spectrum
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Linear scale spectrum
    axes[0, 0].semilogy(freq_positive, psd_positive, 'b-', linewidth=0.8)
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Power Spectral Density')
    axes[0, 0].set_title('Power Spectral Density (Log Scale)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1000)  # Limit to 1kHz
    
    # Log-log scale spectrum
    axes[0, 1].loglog(freq_positive, psd_positive, 'r-', linewidth=0.8)
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Power Spectral Density (Log-Log Scale)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Low frequency region detailed analysis
    low_freq_mask = (freq_positive > 0.1) & (freq_positive < 100)
    freq_low = freq_positive[low_freq_mask]
    psd_low = psd_positive[low_freq_mask]
    
    axes[1, 0].loglog(freq_low, psd_low, 'g-', linewidth=1, label='Measured Data')
    
    # Fit 1/f noise
    if len(freq_low) > 10:
        # Use linear regression to fit log-log data
        log_freq = np.log(freq_low)
        log_psd = np.log(psd_low)
        
        # Linear fitting
        coeffs = np.polyfit(log_freq, log_psd, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Generate fitting line
        fit_psd = np.exp(intercept) * freq_low**slope
        
        axes[1, 0].loglog(freq_low, fit_psd, 'r--', linewidth=2, 
                           label=f'1/f Fit (slope={slope:.2f})')
        
        print(f"\n1/f noise fitting results:")
        print(f"  Slope: {slope:.3f}")
        print(f"  Theoretical 1/f slope: -1.0")
        print(f"  Fit quality: {abs(slope + 1):.3f} (closer to 0 is better)")
    
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('Low Frequency Region 1/f Noise Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spectral density distribution
    axes[1, 1].hist(np.log10(psd_positive), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('log10(Power Spectral Density)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Power Spectral Density Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_1f_noise(frequencies, psd, freq_range=(0.1, 100)):
    """Analyze 1/f noise characteristics"""
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    freq_analysis = frequencies[mask]
    psd_analysis = psd[mask]
    
    if len(freq_analysis) < 10:
        return None, None, None
    
    # Log fitting
    log_freq = np.log(freq_analysis)
    log_psd = np.log(psd_analysis)
    
    coeffs = np.polyfit(log_freq, log_psd, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate R²
    fit_values = intercept + slope * log_freq
    ss_res = np.sum((log_psd - fit_values) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return slope, r_squared, freq_analysis

def detailed_analysis(freq_positive, psd_positive, base_signal):
    """Detailed analysis of 1/f noise"""
    
    # Analyze different frequency ranges
    freq_ranges = [(0.1, 10), (1, 50), (5, 100), (10, 200)]
    
    print("\n1/f noise analysis results:")
    print("=" * 50)
    
    for i, (f_min, f_max) in enumerate(freq_ranges):
        slope, r_squared, freq_used = analyze_1f_noise(freq_positive, psd_positive, (f_min, f_max))
        
        if slope is not None:
            print(f"Frequency range {f_min}-{f_max} Hz:")
            print(f"  Slope: {slope:.3f}")
            print(f"  R²: {r_squared:.3f}")
            print(f"  1/f deviation: {abs(slope + 1):.3f}")
            print(f"  Data points used: {len(freq_used)}")
            print()
    
    # Calculate overall noise level
    noise_level = np.sqrt(np.mean(psd_positive))
    print(f"Overall noise level: {noise_level:.4f}")
    
    # Calculate signal-to-noise ratio
    signal_power = base_signal**2
    noise_power = np.mean(psd_positive)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"Signal-to-noise ratio: {snr:.2f} dB")

def main():
    """Main function"""
    print("Charge noise simulation starting...")
    print("=" * 50)
    
    # 1. Generate charge noise
    t, signal_with_noise, white_noise, pink_noise, base_signal = generate_charge_noise()
    
    # 2. Plot time series
    print("\nPlotting time series...")
    plot_time_series(t, signal_with_noise, white_noise, pink_noise)
    
    # 3. Spectral analysis
    print("\nPerforming spectral analysis...")
    freq_positive, psd_positive = analyze_spectrum(signal_with_noise, 0.05e-3)
    
    # 4. Plot spectrum
    print("\nPlotting spectrum...")
    plot_spectrum(freq_positive, psd_positive)
    
    # 5. Detailed analysis
    print("\nPerforming detailed analysis...")
    detailed_analysis(freq_positive, psd_positive, base_signal)
    
    print("\nSimulation completed!")

if __name__ == "__main__":
    main() 