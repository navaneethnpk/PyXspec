# PyXspec

A Python wrapper for X-ray spectral analysis using PyXSpec.

## Requirements

### System Requirements
- **HEASOFT**: Required for XSPEC/PyXSpec
  - Install from: https://heasarc.gsfc.nasa.gov/lheasoft/
  - Must be initialized (run `heainit` in your shell)

### Python Requirements
- Python >= 3.12
- numpy >= 1.20.0
- matplotlib >= 3.0.0
- pyyaml >= 5.0

## Installation

```bash
# Clone the repository
git clone https://github.com/navaneethnpk/PyXspec.git
cd PyXspec

# Install the package
pip install -e .
```

## Usage

### Prerequisites

Before using PyXspec, ensure:
1. **HEASOFT is initialized** in your environment
2. **Response files are linked** to your PHA file (ARF and RMF paths are set in the PHA header)
3. **PHA file is prepared** (grouping and bad channel flagging already done if needed)

### Basic Example

```python
from pyxspec import XSpecRunner, CommonModels

# Initialize the runner with your PHA file
runner = XSpecRunner(
    pha_file="spectrum.pha",
    out_dir="results",
    en_range=(0.3, 10.0),  # Energy range in keV
    stat_method="chi"       # Fit statistic: "chi" or "cstat"
)

# Create a model (absorbed power-law)
# Note: Get nH value using HEASOFT command: nh <RA> <Dec>
model = CommonModels.absorbed_powerlaw(
    nH=0.05,           # Column density in 10^22 cm^-2
    PhoIndex=2.0,      # Initial photon index
    freeze_nH=True     # Keep nH fixed during fit
)

# Run the fit
results = runner.run_model(model)

# Check results
if results["success"]:
    print(f"Fit succeeded!")
    print(f"Chi-squared: {results['fit_statistics']['chi2red']:.3f}")
    print(f"Flux: {results['flux'][0]:.3e} erg/cm²/s")
else:
    print(f"Fit failed: {results['error']}")
```

### Using Custom Models

```python
from pyxspec import XSpecRunner, ModelManager

runner = XSpecRunner(
    pha_file="spectrum.pha",
    out_dir="results"
)

# Create a custom model
model = ModelManager(
    model="tbabs*(powerlaw+bbody)",
    name="absorbed_pl_bb"
)

# Configure parameters
model.set_parameter("TBabs.nH", 0.05, frozen=True)
model.set_parameter("powerlaw.PhoIndex", 2.0, frozen=False)
model.set_parameter("bbody.kT", 1.0, frozen=False)

# Or set multiple parameters at once
model.set_parameters({
    "TBabs.nH": (0.05, True),           # (value, frozen)
    "powerlaw.PhoIndex": (2.0, False),
    "bbody.kT": (1.0, False)
})

# Run the fit
results = runner.run_model(model)
```

### Advanced Configuration

```python
from pyxspec import XSpecRunner, CommonModels

runner = XSpecRunner(
    pha_file="spectrum.pha",
    out_dir="results",
    en_range=(0.5, 8.0),              # Custom energy range
    stat_method="cstat",               # Use Cash statistic
    plot_xaxis="keV",                  # X-axis units: "keV", "Hz", "channel"
    plot_rebin=(20, 20),              # Plot rebinning (minSig, maxNum)
    plot_types=["eeufspec", "ratio", "delchi"]  # Plot types to generate
)

# Fit multiple models
models = [
    CommonModels.absorbed_powerlaw(nH=0.05),
    CommonModels.absorbed_broken_powerlaw(nH=0.05),
    CommonModels.absorbed_logparabola(nH=0.05)
]

results_list = []
for model in models:
    results = runner.run_model(model)
    results_list.append(results)
    
# Compare models
for res in results_list:
    if res["success"]:
        print(f"{res['model']}: χ²/dof = {res['fit_statistics']['chi2red']:.3f}")
```

## Output Files

For each fit, PyXspec generates:

1. **`{model_name}_fit_results.yaml`**: Fit results including:
   - Model configuration
   - Fit statistics (chi-squared, degrees of freedom, null hypothesis probability)
   - Parameter values and errors
   - Flux and confidence intervals

2. **`{model_name}_xspec.log`**: XSPEC log file with detailed fit output

3. **`{model_name}_plot.ps`**: PostScript plot file

4. **`{model_name}_{plot_type}.csv`**: CSV files for each plot type containing:
   - `x`: X-axis values
   - `dx`: X-axis errors
   - `y`: Y-axis values
   - `dy`: Y-axis errors
   - `model`: Model values (if applicable)

## Utilities

### PHA File Modification with grppha

If you need to modify your PHA file (grouping, bad channel flagging):

```python
from pyxspec.utils import grppha

# Group by minimum 20 counts and flag bad channels
success = grppha(
    pha_file="spectrum.pha",
    commands=[
        "bad 0-29",           # Flag channels 0-29 as bad
        "group min 20",       # Group to minimum 20 counts
    ],
    out_file="spectrum_grp.pha",  # Optional: specify output
)
```

### Custom Plotting

```python
from pyxspec.utils import plot_spectrum
import numpy as np

# Load plot data from CSV
data = np.loadtxt("results/absorbed_powerlaw_eeufspec.csv", 
                  delimiter=",", skiprows=1)

# Prepare data dictionary
main_data = {
    "x": data[:, 0],
    "dx": data[:, 1],
    "y": data[:, 2],
    "dy": data[:, 3],
    "model": data[:, 4]
}

# Load ratio data
ratio_data = np.loadtxt("results/absorbed_powerlaw_ratio.csv",
                        delimiter=",", skiprows=1)

sub_data = {
    "x": ratio_data[:, 0],
    "y": ratio_data[:, 2],
    "dy": ratio_data[:, 3]
}

# Create publication-quality plot
plot_spectrum(
    main_data=main_data,
    sub_data=sub_data,
    plot_name="My Spectrum",
    out_dir="plots",
    scale="log",
    xlabel="Energy (keV)",
    main_ylabel="E²F(E) (keV² photons cm⁻² s⁻¹ keV⁻¹)",
    sub_ylabel="Ratio (Data/Model)",
    plot_ext="pdf"  # Output format: png, pdf, svg, etc.
)
```

## API Reference

### XSpecRunner

Main class for running spectral analysis.

**Parameters:**
- `pha_file`: Path to PHA spectrum file
- `out_dir`: Output directory (default: same as PHA file)
- `en_range`: Energy range tuple (min, max) in keV (default: (0.3, 10.0))
- `stat_method`: Fit statistic "chi" or "cstat" (default: "chi")
- `plot_xaxis`: Plot X-axis units "keV", "Hz", "channel", "angstrom" (default: "keV")
- `plot_rebin`: Plot rebinning (minSig, maxNum) (default: (20, 20))
- `plot_types`: List of plot types (default: ["eeufspec", "ratio"])

**Methods:**
- `run_model(model_config)`: Run fit for a ModelManager configuration

### ModelManager

Model configuration manager.

**Parameters:**
- `model`: PyXSpec model expression (e.g., "tbabs*powerlaw")
- `name`: Descriptive name (optional)

**Methods:**
- `set_parameter(parameter, value, frozen)`: Set a single parameter
- `set_parameters(params_dict)`: Set multiple parameters
- `get_parameter(parameter)`: Get parameter configuration
- `list_parameters()`: List all configured parameters

### CommonModels

Pre-defined common X-ray spectral models.

**Methods:**
- `absorbed_powerlaw(nH, PhoIndex, freeze_nH)`: tbabs\*powerlaw
- `absorbed_broken_powerlaw(nH, PhoIndex1, PhoIndex2, freeze_nH)`: tbabs\*bknpower
- `absorbed_logparabola(nH, alpha, beta, freeze_nH)`: tbabs\*logpar


## Acknowledgments

This package is built on top of:
- **XSPEC**: X-ray spectral fitting package by NASA/HEASARC
- **PyXSpec**: Python interface to XSPEC