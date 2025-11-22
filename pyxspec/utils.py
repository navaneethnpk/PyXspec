"""
Utility functions for X-ray data preparation and manipulation.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from .logging import setup_logger

# Module logger
logger = setup_logger(__name__)


def grppha(
    pha_file: str,
    commands: List[str],
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    overwrite: bool = False,
) -> bool:
    """
    Run GRPPHA to modify a PHA file with custom commands.

    Parameters
    ----------
    pha_file : str
        Input PHA filename (e.g., "spectrum.pi" or "spectrum.pha")
    commands : List[str]
        List of GRPPHA commands to execute (e.g., ["bad 0-29", "group min 20"])
    output_file : str, optional
        Output PHA filename. If None, creates filename with '_grp' suffix.
    log_file : str, optional
        Log filename. If None, creates filename with '_grppha.log' suffix.
    overwrite : bool, optional
        If True, overwrites the input file instead of creating new output file.
        Default is False.

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
    if overwrite:
        output_filename = pha_file
        logger.info(f"Overwrite mode: will modify {pha_file} in place")
    elif output_file is not None:
        output_filename = output_file
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
    logger.debug(f"GRPPHA commands:\n{grppha_cmd}")

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
    except subprocess.TimeoutExpired:
        cli_output = "GRPPHA process timed out"
        success = False
        logger.error(f"GRPPHA timed out for {pha_file}")
    except FileNotFoundError:
        cli_output = "GRPPHA command not found. Ensure HEASOFT is installed."
        success = False
        logger.error("GRPPHA not found in PATH")

    # Write log
    with open(log_filename, "w") as f:
        f.write(cli_output)

    logger.debug(f"GRPPHA log written to {log_filename}")

    return success
