# Charge Noise Simulation and Analysis

This repository contains scripts for simulating and analyzing charge noise with different spectral characteristics, particularly focusing on pink noise (1/f noise) analysis.

## Scripts Overview

### 1. `charge_noise_simulation.py`

**Purpose**: Original charge noise simulation script that generates and analyzes different types of noise.

**Features**:

- Generates white noise, pink noise (1/f), brown noise, and periodic noise
- Combines different noise components to simulate realistic charge noise
- Performs time series analysis and spectral analysis
- Uses both Welch method and FFT for spectral analysis

**Output Files**:

- `time_series_analysis.png`: Time series plots and amplitude distribution
- `spectrum_analysis_Welch.png`: Power spectral density using Welch method
- `spectrum_analysis_FFT.png`: Power spectral density using FFT method

### 2. `pink_noise_simulation/pink_noise_analysis.py`

**Purpose**: Advanced pink noise analysis with different power law exponents (α values).

**Features**:

- Generates pink noise with different α values (0.5, 1.0, 1.5, 2.0)
- Normalizes noise to standard deviation = 1
- Analyzes noise levels at specific frequencies (0.1, 1.0, 10.0 Hz)
- Plots amplitude spectral power vs frequency
- Compares theoretical vs measured 1/f spectra

**Key Analysis**:

- **Power Law Generation**: Creates noise with 1/f^α spectral density
- **Standard Deviation Normalization**: All noise signals have s.d. = 1
- **Frequency Analysis**: Analyzes noise levels at 0.1, 1, and 10 Hz
- **Spectral Power Analysis**: Plots √PSD vs frequency to check noise per √Hz

**Output Files**:

- `figures/pink_noise_comparison.png`: Time series and distribution comparison
- `figures/pink_noise_spectrum_Welch.png`: Spectral analysis using Welch method
