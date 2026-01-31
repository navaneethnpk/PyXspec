"""
PyXSpec Wrapper for X-ray Spectral Analysis.

Generic wrapper for PyXSpec that works with data from multiple X-ray missions:
Swift-XRT, NICER, NuSTAR, XMM-Newton etc.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
except ImportError as e:
    raise ImportError(
        "PyXSpec not available. Install HEASOFT to use this module."
    ) from e


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
    plot_xaxis : str, optional
        X-axis units for plots: "keV", "Hz", "channel", "angstrom" (default: "keV")
    plot_rebin : tuple of int, optional
        Rebinning specification as (minSig, maxNum) for plot display (default: "20 20")
    plot_types : list of str, optional
        List of plot types to generate (default: ["eeufspec", "ratio"])

    Attributes
    ----------
    pha_file : Path
        Path to the PHA file
    out_dir : Path
        Output directory for all results (YAML, CSV, PostScript plots)
    en_range : tuple of float
        Energy range (min, max) in keV for analysis
    stat_method : str
        Fit statistic method (chi, or cstat)
    plot_xaxis : str
        X-axis units for plots
    plot_rebin : tuple of int
        Rebinning parameters (minSig, maxNum)
    plot_types : list of str
        List of plot types to generate

    Raises
    ------
    FileNotFoundError
        If the specified PHA file does not exist
    """

    def __init__(
        self,
        pha_file: Union[str, Path],
        out_dir: Optional[Union[str, Path]] = None,
        en_range: Tuple[float, float] = (0.3, 10.0),
        stat_method: str = "chi",
        plot_xaxis: str = "keV",
        plot_rebin: Tuple[int, int] = (20, 20),
        plot_types: Optional[list] = None,
    ):
        """Initialize XSpec runner."""
        # Validate and set PHA file path
        self.pha_file = Path(pha_file).expanduser().resolve()
        if not self.pha_file.exists():
            raise FileNotFoundError(f"PHA file not found: {self.pha_file}")

        # Set output directory
        if out_dir is None:
            self.out_dir = self.pha_file.parent
        else:
            self.out_dir = Path(out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Store analysis parameters
        self.en_range = en_range
        self.stat_method = stat_method

        # Store plot configuration
        self.plot_xaxis = plot_xaxis
        self.plot_rebin = plot_rebin
        self.plot_types = (
            plot_types if plot_types is not None else ["eeufspec", "ratio"]
        )

        logger.debug(f"Initialized XSpecRunner for {self.pha_file}")
        logger.debug(f"Energy range: {en_range[0]} - {en_range[1]} keV")
        logger.debug(
            f"Plot config: axis={plot_xaxis}, rebin={plot_rebin}, types={self.plot_types}"
        )
        logger.debug(f"Output directory: {self.out_dir}")

    def _setup_xspec(self):
        """Setup XSPEC environment and clear any existing data/models."""
        AllData.clear()
        AllModels.clear()

        # Configure output verbosity
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

        # Ignore bad channels flagged in the PHA file
        AllData.ignore("bad")

        # Apply energy range: ignore data outside the specified range
        ignore_range = f"**-{self.en_range[0]},{self.en_range[1]}-**"
        spectrum.ignore(ignore_range)

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

        # Apply parameter settings from configuration
        for params, config in model_config.parameters.items():
            component_name, param_name = params.split(".")

            try:
                # Access the component and parameter
                component = getattr(model, component_name)
                param = getattr(component, param_name)

                # Set parameter value and freeze status
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

        logger.debug("Starting spectral fit")

        # Configure fit parameters
        Fit.statMethod = self.stat_method
        Fit.nIterations = 100
        Fit.query = "yes"

        # Renormalize model to data and perform fit
        Fit.renorm()
        Fit.perform()
        Fit.show()

    def _extract_fit_statistics(self) -> Dict:
        """
        Extract fit statistics from the completed fit.

        Returns
        -------
        dict
            Fit statistics including:
            - statistic: The fit statistic value
            - testStatistic: Test statistic value
            - dof: Degrees of freedom
            - chi2red: Reduced chi-squared (testStatistic/dof)
            - statMethod: Statistical method used (chi, cstat, etc.)
            - statTest: Statistical test used
            - nullhyp: Null hypothesis probability
            - nVarPars: Number of variable parameters
        """
        fit_stats = {
            "statistic": Fit.statistic,
            "testStatistic": Fit.testStatistic,
            "dof": Fit.dof,
            "chi2red": Fit.testStatistic / Fit.dof if Fit.dof > 0 else np.nan,
            "statMethod": Fit.statMethod,
            "statTest": Fit.statTest,
            "nullhyp": Fit.nullhyp,
            "nVarPars": Fit.nVarPars,
        }

        logger.debug(
            f"Fit statistics: {fit_stats['statMethod']} = {fit_stats['statistic']:.2f}"
        )
        logger.debug(
            f"Test statistic/dof = {fit_stats['testStatistic']:.2f}/{fit_stats['dof']} = {fit_stats['chi2red']:.3f}"
        )
        logger.debug(f"Null hypothesis probability: {fit_stats['nullhyp']:.4f}")
        logger.debug(f"Number of variable parameters: {fit_stats['nVarPars']}")

        return fit_stats

    def _extract_parameter_errors(self, model: Model) -> Dict:
        """
        Extract parameter values and errors from the fitted model.

        Parameters
        ----------
        model : Model
            PyXSpec Model object

        Returns
        -------
        dict
            Dictionary of parameter information with keys as parameter indices.
            Each entry contains:
            - name: Parameter name
            - value: Best-fit value
            - sigma: 1-sigma error
            - lower_bound: Lower error bound
            - upper_bound: Upper error bound
            - frozen: Whether parameter was frozen
        """
        parameters = {}

        # Iterate through all model parameters
        for i in range(1, model.nParameters + 1):
            # Calculate confidence intervals (90% by default)
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

        return parameters

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
        # Calculate flux with 90% confidence errors
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
            Dictionary with keys for each plot type containing plot data.
            Each plot type contains arrays for x, dx, y, dy, and model (if applicable).
        """
        # Configure plot settings
        Plot.xAxis = self.plot_xaxis
        Plot.xLog = True
        Plot.yLog = True
        Plot.setRebin(self.plot_rebin[0], self.plot_rebin[1])

        # Create PostScript plot
        Plot.device = f"{self.out_dir}/{model_config.name}_plot.ps"
        Plot(*self.plot_types)

        # Switch to null device for data extraction
        Plot.device = "/null"

        # Plot types that have model data
        model_plot_types = [
            "data",
            "ldata",
            "eemodel",
            "eufspec",
            "eeufspec",
            "model",
            "ufspec",
            "counts",
            "lcounts",
        ]

        # Extract data for each plot type
        plot_data = {}
        for plot_type in self.plot_types:
            logger.debug(f"Extracting data for plot type: {plot_type}")
            Plot(plot_type)

            # Get basic plot arrays from XSPEC
            x_data = Plot.x()
            x_err = Plot.xErr()
            y_data = Plot.y()
            y_err = Plot.yErr()

            # Check if this plot type should have model data
            has_model = plot_type in model_plot_types

            # Build plot_data dictionary and CSV array
            if has_model:
                model_data = Plot.model()
                plot_data[plot_type] = {
                    "x": x_data,
                    "dx": x_err,
                    "y": y_data,
                    "dy": y_err,
                    "model": model_data,
                }
                data_array = np.column_stack([x_data, x_err, y_data, y_err, model_data])
                header = "x,dx,y,dy,model"
            else:
                plot_data[plot_type] = {
                    "x": x_data,
                    "dx": x_err,
                    "y": y_data,
                    "dy": y_err,
                }
                data_array = np.column_stack([x_data, x_err, y_data, y_err])
                header = "x,dx,y,dy"

            # Save to CSV file
            csv_file = self.out_dir / f"{model_config.name}_{plot_type}.csv"
            np.savetxt(
                csv_file,
                data_array,
                delimiter=",",
                header=header,
                comments="",
            )
            logger.debug(f"Saved {plot_type} data to {csv_file}")

        return plot_data

    def _cleanup(self):
        """Clean up XSPEC environment and reset plot settings."""
        # Reset plot settings to defaults
        Plot.device = "/null"
        Plot.xAxis = "keV"
        Plot.xLog = False
        Plot.yLog = False

        # AllData.clear()
        # AllModels.clear()

        # Close log file
        Xset.closeLog()

        logger.debug("XSpec environment cleaned up.")

    def run_model(
        self,
        model_config: ModelManager,
    ) -> Dict:
        """
        Run XSPEC analysis for a single model configuration.

        Parameters
        ----------
        model_config : ModelManager
            Model configuration to fit

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
            - plot_data : dict
                Plot data arrays for each plot type
            - success : bool
                Whether fit succeeded
            - error : str
                Error message (if failed)
        """
        pha_name = self.pha_file.name
        try:
            # Setup XSPEC environment
            self._setup_xspec()
            log_file = self.out_dir / f"{model_config.name}_xspec.log"
            Xset.openLog(str(log_file))
            logger.info(f"Analysis started: PHA={pha_name}, Model={model_config.name}")

            # Load and prepare data
            spectrum = self._load_data()

            # Create and configure model
            model = self._create_model(model_config)

            # Perform spectral fit
            self._perform_fit()

            # Extract fit, parameters and flux results
            fit_stats = self._extract_fit_statistics()
            parameters = self._extract_parameter_errors(model)
            flux = self._calculate_flux(spectrum)

            # Save results to YAML
            self._save_results(model_config, fit_stats, parameters, flux)

            # Extract and save plot data to CSV
            plot_data = self._extract_plot_data(model_config)

            logger.info(
                f"Analysis completed: PHA={pha_name}, Model={model_config.name}"
            )

            return {
                "model": model_config.name,
                "fit_statistics": fit_stats,
                "parameters": parameters,
                "flux": flux,
                "plot_data": plot_data,
                "success": True,
            }

        except Exception as e:
            logger.info(
                f"Analysis failed: PHA={pha_name}, Model={model_config.name}: {e}",
                exc_info=True,
            )
            return {
                "model": model_config.name,
                "error": str(e),
                "success": False,
            }

        finally:
            # Always clean up, even if analysis failed
            self._cleanup()
