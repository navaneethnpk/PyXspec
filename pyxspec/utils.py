"""
Utility functions for X-ray data preparation and analysis.
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np

from .logging import setup_logger

# Module logger
logger = setup_logger(__name__)


def grppha(
    pha_file: str,
    commands: List[str],
    out_file: Optional[str] = None,
    log_file: Optional[str] = None,
) -> bool:
    """
    Run GRPPHA to modify a PHA file with custom commands.

    Parameters
    ----------
    pha_file : str
        Input PHA filename (e.g., "spectrum.pi" or "spectrum.pha")
    commands : List[str]
        List of GRPPHA commands to execute (e.g., ["bad 0-29", "group min 20"])
    out_file : str, optional
        Output PHA filename. If None, creates filename with '_grp' suffix.
    log_file : str, optional
        Log filename. If None, creates filename with '_grppha.log' suffix.

    Returns
    -------
    bool
        True if GRPPHA ran successfully, False otherwise
    """

    # Check if input file exists
    pha_path = Path(pha_file)
    if not pha_path.exists():
        logger.error(f"PHA file not found: {pha_file}")
        return False

    # Determine output and log filenames
    file_extension = pha_path.suffix
    if out_file is not None:
        output_filename = out_file
        if output_filename == pha_file:
            logger.info(f"Overwrite mode: will modify {pha_file} in place")
    else:
        output_filename = pha_file.replace(file_extension, f"_grp{file_extension}")
    if log_file is not None:
        log_filename = log_file
    else:
        log_filename = pha_file.replace(file_extension, "_grppha.log")

    # Build GRPPHA command
    cmd_lines = [
        pha_file,
        f"!{output_filename}",
    ]
    cmd_lines.extend(commands)  # Add user's commands
    cmd_lines.append("exit")  # Always end with exit
    grppha_cmd = "\n".join(cmd_lines)

    logger.info(f"Running GRPPHA on {pha_file}")
    logger.debug(f"GRPPHA commands: [{grppha_cmd.replace('\n', ', ')}]")

    try:
        grppha_process = subprocess.run(
            ["grppha"],
            input=grppha_cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        cli_output = grppha_process.stdout + grppha_process.stderr
        success = True
        logger.info(f"GRPPHA completed successfully. Output: {output_filename}")
    except subprocess.CalledProcessError as e:
        cli_output = (e.stdout or "") + (e.stderr or "")
        success = False
        logger.error(f"GRPPHA failed for {pha_file}: {e}")

    # Write log
    with open(log_filename, "w") as f:
        f.write(cli_output)

    logger.debug(f"GRPPHA log written to {log_filename}")

    return success


def plot_spectrum(
    main_data: Dict[str, np.ndarray],
    sub_data: Optional[Dict[str, np.ndarray]] = None,
    plot_name: str = "Spectrum",
    out_dir: Union[str, Path] = None,
    scale: str = "log",
    xlabel: str = "",
    main_ylabel: str = "",
    sub_ylabel: str = "",
    plot_ext: str = "png",
):
    """
    Create publication-quality spectrum plot with optional subplot panel.

    Generates a spectrum plot with data points, optional model overlay, and
    optional subplot (for ratio, residual, or delchi). Automatically handles
    log/linear scaling, error bars, and saves to file.

    Parameters
    ----------
    main_data : dict
        Dictionary containing main spectrum data with keys:
        - 'x' : X-axis values (required)
        - 'y' : Y-axis values (required)
        - 'dx' : X-axis errors (optional)
        - 'dy' : Y-axis errors (optional)
        - 'model' : Model values for overlay (optional)
    sub_data : dict, optional
        Dictionary for subplot panel (ratio/residual/delchi) with keys:
        - 'x' : X-axis values (required)
        - 'y' : Y-axis values (required)
        - 'dy' : Y-axis errors (optional)
        If None, only main panel is plotted (default: None)
    plot_name : str, optional
        Name used for plot title and output filename (default: "Spectrum")
    out_dir : str or Path, optional
        Output directory path. If None, uses current working directory (default: None)
    scale : str, optional
        Axis scaling: "log" for log-log, "lin" for linear-linear (default: "log")
    xlabel : str, optional
        X-axis label (default: "")
    main_ylabel : str, optional
        Y-axis label for main panel (default: "")
    sub_ylabel : str, optional
        Y-axis label for subplot panel (default: "")
    plot_ext : str, optional
        Output file extension/format (e.g., "png", "pdf", "svg") (default: "png")

    Returns
    -------
    None
        Plot is saved to file. No return value.

    Raises
    ------
    ValueError
        If main_data does not contain required 'x' and 'y' keys
    """
    # Validate required keys
    if "x" not in main_data or "y" not in main_data:
        raise ValueError("main_data dictionary must contain 'x' and 'y' keys")

    # Extract main panel data
    main_x = main_data["x"]
    main_y = main_data["y"]
    main_dx = main_data.get("dx", None)
    main_dy = main_data.get("dy", None)
    main_model = main_data.get("model", None)

    # Determine if we need a subplot
    has_sub = sub_data is not None and "x" in sub_data and "y" in sub_data

    # Create figure
    if has_sub:
        fig, (ax_main, ax_sub) = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(8, 6))

    # Main spectrum plot
    ax_main.errorbar(
        main_x,
        main_y,
        xerr=main_dx,
        yerr=main_dy,
        fmt="o",
        markersize=4,
        label="Data",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=1,
    )

    # Plot model if provided
    if main_model is not None:
        ax_main.plot(
            main_x,
            main_model,
            label="Model",
            color="red",
            linestyle="-",
            linewidth=1.5,
        )

    # Set scales for main panel
    scale_lower = scale.lower()
    if scale_lower == "log":
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
    elif scale_lower == "lin":
        ax_main.set_xscale("linear")
        ax_main.set_yscale("linear")

    # Labels and styling
    if not has_sub and xlabel:
        ax_main.set_xlabel(xlabel, fontsize=16)
    if main_ylabel:
        ax_main.set_ylabel(main_ylabel, fontsize=16)

    if main_model is not None:
        ax_main.legend(
            loc="best", fontsize=12, frameon=True, fancybox=False, borderpad=0.6
        )

    ax_main.grid(True, alpha=0.1, which="both")
    ax_main.tick_params(
        labelsize=12,
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
    )
    ax_main.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True
    )

    # Subplot (ratio/residual/delchi) if requested
    if has_sub:
        sub_x = sub_data["x"]
        sub_y = sub_data["y"]
        sub_dy = sub_data.get("dy", None)

        ax_sub.errorbar(
            sub_x,
            sub_y,
            yerr=sub_dy,
            fmt="o",
            markersize=2,
            color="black",
            ecolor="gray",
            elinewidth=1,
            capsize=1,
        )

        # Add reference line: y=1 for ratio, y=0 for residuals
        if sub_ylabel and "ratio" in sub_ylabel.lower():
            ax_sub.axhline(1.0, color="red", linestyle="-", linewidth=1.0)
        else:
            ax_sub.axhline(0.0, color="red", linestyle="-", linewidth=1.0)

        # X-axis scale matches main panel
        if scale_lower == "log":
            ax_sub.set_xscale("log")

        if xlabel:
            ax_sub.set_xlabel(xlabel, fontsize=16)
        if sub_ylabel:
            ax_sub.set_ylabel(sub_ylabel, fontsize=16)

        ax_sub.tick_params(
            labelsize=12,
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
        )
        ax_sub.tick_params(
            axis="both", which="minor", direction="in", top=True, right=True
        )
        ax_sub.grid(True, alpha=0.2, which="both")

        # Fix auto-scaling for subplot
        ax_sub.relim()
        ax_sub.autoscale_view()

    # Fix auto-scaling for main panel
    ax_main.relim()
    ax_main.autoscale_view()

    fig.suptitle(plot_name, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"{plot_name.replace(' ', '_')}.{plot_ext}"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
