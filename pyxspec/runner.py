"""
PyXSpec Wrapper for X-ray Spectral Analysis.

Generic wrapper for PyXSpec that works with data from multiple X-ray missions:
Swift-XRT, NICER, NuSTAR, Chandra, XMM-Newton etc.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .models import ModelManager
from .logging import setup_logger

# Module logger
logger = setup_logger(__name__)

# Import PyXSpec components
try:
    from xspec import (
        AllData,
        AllModels,
        Fit,
        Model,
        Plot,
        Spectrum,
        Xset,
    )

    XSPEC_AVAILABLE = True
except ImportError:
    XSPEC_AVAILABLE = False
    logger.warning("PyXSpec not available. Install HEASOFT to use this module.")


class XSpecRunner:
    """
    Generic wrapper for PyXSpec spectral fitting.

    Provides a clean interface to PyXSpec for fitting X-ray spectra.
    Handles data loading, model creation, fitting, flux calculation, and result output.

    Parameters
    ----------
    pha_file : str or Path
        Path to the PHA spectrum file
    out_dir : str or Path, optional
        Output directory for results (default: same as PHA file directory)
    en_range : tuple of float, optional
        Energy range for fitting (min, max) in keV (default: (0.3, 10.0))
    stat_method : str, optional
        Fit statistic method: "chi", "cstat", or "pgstat" (default: "chi")

    Attributes
    ----------
    pha_file : Path
        Path to the PHA file
    out_dir : Path
        Output directory
    en_range : tuple
        Energy range for analysis
    stat_method : str
        Fit statistic method
    """

    def __init__(
        self,
        pha_file: Union[str, Path],
        out_dir: Optional[Union[str, Path]] = None,
        en_range: Tuple[float, float] = (0.3, 10.0),
        stat_method: str = "chi",
    ):
        """Initialize XSpec runner."""
        if not XSPEC_AVAILABLE:
            raise ImportError(
                "PyXSpec is not available. Please install HEASOFT with PyXSpec support."
            )

        self.pha_file = Path(pha_file).expanduser().resolve()
        if not self.pha_file.exists():
            raise FileNotFoundError(f"PHA file not found: {self.pha_file}")

        if out_dir is None:
            self.out_dir = self.pha_file.parent
        else:
            self.out_dir = Path(out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.en_range = en_range
        self.stat_method = stat_method

        logger.debug(f"Initialized XSpecRunner for {self.pha_file}")
        logger.debug(f"Energy range: {en_range[0]} - {en_range[1]} keV")
        logger.debug(f"Output directory: {self.out_dir}")

    def _setup_xspec(self):
        """Setup XSPEC environment and clear any existing data/models."""
        AllData.clear()
        AllModels.clear()

        Xset.chatter = 0
        Xset.logChatter = 20

        logger.debug("XSpec environment initialized")

    def _load_data(self) -> Spectrum:
        """
        Load the spectrum data and apply energy cuts.

        Returns
        -------
        Spectrum
            PyXSpec Spectrum object with energy filters applied
        """
        logger.debug(f"Loading spectrum: {self.pha_file}")
        spectrum = Spectrum(str(self.pha_file))

        # Ignore bad channels
        AllData.ignore("bad")

        # Apply energy range
        ignore_range = f"**-{self.en_range[0]},{self.en_range[1]}-**"
        spectrum.ignore(ignore_range)

        logger.debug(
            f"Applied energy filter: {self.en_range[0]}-{self.en_range[1]} keV"
        )

        return spectrum

    def _create_model(self, model_config: ModelManager) -> Model:
        """
        Create and configure PyXSpec model from ModelManager.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration with expression and parameters

        Returns
        -------
        Model
            Configured PyXSpec Model object

        Raises
        ------
        ValueError
            If parameter path is invalid for the model
        """
        logger.debug(f"Creating model: {model_config.expression}")
        model = Model(model_config.expression)

        # Apply parameter settings
        for params, config in model_config.parameters.items():
            component_name, param_name = params.split(".")

            try:
                component = getattr(model, component_name)
                param = getattr(component, param_name)

                param.values = config["value"]
                param.frozen = config["frozen"]

                logger.debug(
                    f"Set {params} = {config['value']} (frozen = {config['frozen']})"
                )
            except AttributeError as e:
                logger.error(
                    f"Failed to set parameter '{params}'. "
                    f"Check if component/parameter names are correct in the model."
                )
                raise ValueError(
                    f"Invalid parameter path '{params}' for model '{model.expression}'"
                ) from e

        return model

    def _perform_fit(self):
        """Perform the spectral fit using configured statistics method."""

        logger.debug("Starting spectral fit...")

        Fit.statMethod = self.stat_method
        Fit.nIterations = 100
        Fit.query = "yes"

        Fit.perform()
        Fit.show()

    def _extract_fit_results(self, model: Model) -> Tuple[Dict, Dict]:
        """
        Extract fit statistics and parameter values.

        Parameters
        ----------
        model : Model
            PyXSpec Model object

        Returns
        -------
        tuple of dict
            (fit_statistics, parameters)
            - fit_statistics: chi2, dof, chi2red
            - parameters: parameter values, errors, and metadata
        """
        chi = Fit.statistic
        dof = Fit.dof
        chi2red = Fit.testStatistic / dof if dof > 0 else np.nan

        fit_stats = {
            "statistic": chi,
            "dof": dof,
            "chi2red": chi2red,
        }

        # Extract parameter information
        parameters = {}
        for i in range(1, model.nParameters + 1):
            Fit.error(f"{i}")
            param = AllModels(1)(i)

            param_info = {
                "name": param.name,
                "value": param.values[0],
                "sigma": param.sigma,
                "lower_bound": param.error[0],
                "upper_bound": param.error[1],
                "frozen": param.frozen,
            }
            parameters[i] = param_info

            logger.debug(
                f"Parameter {i} ({param.name}): {param.values[0]:.3e} ± {param.sigma:.3e}"
            )
        logger.debug(f"Fit statistics: χ²/dof = {chi:.2f}/{dof} = {chi / dof:.2f}")

        return fit_stats, parameters

    def _calculate_flux(self, spectrum: Spectrum) -> Tuple[float, float, float]:
        """
        Calculate flux in the specified energy range.

        Parameters
        ----------
        spectrum : Spectrum
            PyXSpec Spectrum object

        Returns
        -------
        tuple of float
            (flux, lower_error, upper_error) in erg/cm²/s
        """
        flux_str = f"{self.en_range[0]} {self.en_range[1]} error 100 90"
        AllModels.calcFlux(flux_str)

        flux = spectrum.flux
        logger.debug(f"Calculated flux: {flux[0]:.3e} ({flux[1]:.3e}, {flux[2]:.3e})")

        return flux

    def _save_results(
        self,
        model_config: ModelManager,
        fit_stats: Dict,
        parameters: Dict,
        flux: Tuple[float, float, float],
    ):
        """
        Save fit results to YAML file.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration
        fit_stats : dict
            Fit statistics (chi2, dof, chi2red)
        parameters : dict
            Parameter values and errors
        flux : tuple of float
            Flux and errors
        """
        results_dict = {
            "model": {
                "name": model_config.name,
                "expression": model_config.expression,
            },
            "fit_statistics": fit_stats,
            "parameters": parameters,
            "flux": {
                "value": flux[0],
                "lower_bound": flux[1],
                "upper_bound": flux[2],
                "en_range_keV": list(self.en_range),
            },
            "input_files": {
                "spectrum": str(self.pha_file),
            },
        }

        output_file = self.out_dir / f"{model_config.name}_fit_results.yaml"
        with open(output_file, "w") as f:
            yaml.dump(results_dict, f, sort_keys=False, default_flow_style=False)

        logger.debug(f"Results saved to {output_file}")

    def _extract_plot_data(self, model_config: ModelManager) -> Dict[str, Dict]:
        """
        Extract plot data from PyXSpec and save to CSV files.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration

        Returns
        -------
        dict
            Dictionary with 'spectrum' and 'ratio' keys containing plot data
        """
        # Configure plot settings
        Plot.xAxis = "Hz"
        Plot.xLog = True
        Plot.yLog = True
        Plot.setRebin(5, 20)

        # Create PostScript plot
        Plot.device = f"{self.out_dir}/{model_config.name}_plot.ps"
        Plot("eeufspec", "ratio")

        # Switch to null device for data extraction
        Plot.device = "/null"

        # Extract spectrum data
        Plot("eeufspec")
        spec_data = np.column_stack(
            [Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr(), Plot.model()]
        )

        # Save to CSV
        spec_file = self.out_dir / f"{model_config.name}_spectrum.csv"
        np.savetxt(
            spec_file,
            spec_data,
            delimiter=",",
            header="energy,energy_err,flux,flux_err,model",
            comments="",
        )
        logger.debug(f"Spectrum data saved to {spec_file}")

        # Extract ratio data
        Plot("ratio")
        ratio_data = np.column_stack([Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr()])

        # Save to CSV
        ratio_file = self.out_dir / f"{model_config.name}_ratio.csv"
        np.savetxt(
            ratio_file,
            ratio_data,
            delimiter=",",
            header="energy,energy_err,ratio,ratio_err",
            comments="",
        )
        logger.debug(f"Ratio data saved to {ratio_file}")

        spectrum_dict = {
            "x": spec_data[:, 0],
            "dx": spec_data[:, 1],
            "y": spec_data[:, 2],
            "dy": spec_data[:, 3],
            "model": spec_data[:, 4],
        }

        ratio_dict = {
            "x": ratio_data[:, 0],
            "dx": ratio_data[:, 1],
            "y": ratio_data[:, 2],
            "dy": ratio_data[:, 3],
        }

        return {
            "spectrum": spectrum_dict,
            "ratio": ratio_dict,
        }

    def _create_plots(self, model_config: ModelManager, data: Dict):
        """
        Create publication-quality spectrum and residual plots.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration
        data : dict
            Plot data from _extract_plot_data()
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Spectrum plot
        ax1.errorbar(
            data["spectrum"]["x"],
            data["spectrum"]["y"],
            yerr=data["spectrum"]["dy"],
            fmt="o",
            markersize=4,
            label="Data",
            color="black",
            ecolor="gray",
            elinewidth=1,
            capsize=1,
        )
        ax1.plot(
            data["spectrum"]["x"],
            data["spectrum"]["model"],
            label="Model",
            color="red",
            linestyle="-",
            linewidth=1.0,
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$\nu F \nu$ (erg cm$^{-2}$ s$^{-1}$)", fontsize=12)
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.2, which="both")
        ax1.tick_params(
            labelsize=10,
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
        )
        ax1.tick_params(
            axis="both", which="minor", direction="in", top=True, right=True
        )

        x_min = data["spectrum"]["x"].min()
        x_max = data["spectrum"]["x"].max()
        ax1.set_xlim(x_min * 0.8, x_max * 1.2)  # Add 20% padding

        # Ratio plot
        ax2.errorbar(
            data["ratio"]["x"],
            data["ratio"]["y"],
            yerr=data["ratio"]["dy"],
            fmt="o",
            markersize=2,
            color="black",
            ecolor="gray",
            elinewidth=1,
            capsize=1,
        )
        ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
        ax2.set_xscale("log")
        ax2.set_xlabel("Energy (Hz)", fontsize=12)
        ax2.set_ylabel("Ratio", fontsize=12)
        ax2.tick_params(
            labelsize=10,
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
        )
        ax2.tick_params(
            axis="both", which="minor", direction="in", top=True, right=True
        )

        # Title
        fig.suptitle(
            f"Spectral Fit: {model_config.name} ({self.pha_file.name})",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save
        plot_file = self.out_dir / f"{model_config.name}_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.debug(f"Plot saved to {plot_file}")

    def _cleanup(self):
        """Clean up XSPEC environment and reset plot settings."""
        Plot.device = "/null"
        Plot.xAxis = "keV"
        Plot.xLog = False
        Plot.yLog = False

        AllData.clear()
        Xset.closeLog()

        logger.debug("XSpec environment cleaned up")

    def run_model(
        self,
        model_config: ModelManager,
        create_plots: bool = True,
    ) -> Dict:
        """
        Run XSPEC analysis for a single model configuration.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration to fit
        create_plots : bool, optional
            Whether to create plots (default: True)

        Returns
        -------
        dict
            Results dictionary containing:
            - model : str
                Model name
            - fit_statistics : dict
                Chi-squared, DOF, reduced chi-squared
            - parameters : dict
                Parameter values and errors
            - flux : tuple
                Flux and errors in erg/cm²/s
            - success : bool
                Whether fit succeeded
            - error : str
                Error message (if failed)
        """
        try:
            # Setup
            self._setup_xspec()
            log_file = self.out_dir / f"{model_config.name}_xspec.log"
            Xset.openLog(str(log_file))

            logger.debug(f"Starting analysis with model: {model_config.name}")

            # Load data
            spectrum = self._load_data()

            # Create and configure model
            model = self._create_model(model_config)

            # Perform fit
            self._perform_fit()

            # Extract results
            fit_stats, parameters = self._extract_fit_results(model)

            # Calculate flux
            flux = self._calculate_flux(spectrum)

            # Save results
            self._save_results(model_config, fit_stats, parameters, flux)

            # Extract and save plot data
            plot_data = self._extract_plot_data(model_config)

            # Create plots if requested
            self._create_plots(model_config, plot_data)

            logger.debug(f"Xspec analysis completed for model: {model_config.name}")

            return {
                "model": model_config.name,
                "fit_statistics": fit_stats,
                "parameters": parameters,
                "flux": flux,
                "success": True,
            }

        except Exception as e:
            logger.error(
                f"Analysis failed for model {model_config.name}: {e}", exc_info=True
            )
            return {
                "model": model_config.name,
                "error": str(e),
                "success": False,
            }

        finally:
            self._cleanup()
