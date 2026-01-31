"""
Model configuration and management for PyXSpec.
"""

from typing import Dict, List, Optional, Tuple, Union

from .logging import setup_logger

# Module logger
logger = setup_logger(__name__)


class ModelManager:
    """
    Manager for PyXSpec model configurations.

    Parameters
    ----------
    model : str
        PyXSpec model expression (e.g., "tbabs*powerlaw", "phabs*(powerlaw+bbody)")
    name : str, optional
        Descriptive name for the model (used for output files)

    Attributes
    ----------
    expression : str
        The PyXSpec model expression
    name : str
        Model name
    parameters : dict
        Parameter settings {component.param: {"value": ..., "frozen": ...}}
    """

    def __init__(self, model: str, name: Optional[str] = None):
        """Initialize model manager."""
        self.expression = model
        self.name = name or self._generate_name(model)
        self.parameters = {}

        logger.debug(f"Created ModelManager: {self.name} ({self.expression})")

    @staticmethod
    def _generate_name(expression: str) -> str:
        """Generate a readable name from model expression."""
        name = expression.replace("*", "_")
        name = name.replace("+", "_")
        name = name.replace("(", "").replace(")", "")
        return name

    def set_parameter(
        self,
        parameter: str,
        value: float,
        frozen: bool = False,
    ):
        """
        Set a model parameter value and freeze state.

        Parameters
        ----------
        parameter : str
            Full parameter path (e.g., "TBabs.nH", "powerlaw.PhoIndex")
        value : float
            Initial parameter value
        frozen : bool, optional
            Whether to freeze the parameter (default: False)
        """
        param_config = {
            "value": value,
            "frozen": frozen,
        }

        self.parameters[parameter] = param_config

        logger.debug(f"Set parameter {parameter} = {value} (frozen={frozen})")

    def set_parameters(self, params_dict: Dict[str, Union[Tuple, Dict]]):
        """
        Set multiple parameters at once.

        Parameters
        ----------
        params_dict : dict
            Dictionary mapping parameter paths to values. Values can be:
            - tuple: (value, frozen)
            - dict: {"value": val, "frozen": bool}
        """
        for param_path, config in params_dict.items():
            if isinstance(config, tuple):
                value, frozen = config
                self.set_parameter(param_path, value, frozen)
            elif isinstance(config, dict):
                self.set_parameter(param_path, **config)
            else:
                raise ValueError(f"Invalid parameter configuration for {param_path}")

    def get_parameter(self, parameter: str) -> Optional[dict]:
        """Get configuration for a specific parameter."""
        return self.parameters.get(parameter)

    def list_parameters(self) -> List[str]:
        """List all configured parameters."""
        return list(self.parameters.keys())

    def copy(self):
        """Create a copy of this model configuration."""
        new_model = ModelManager(self.expression, self.name)
        new_model.parameters = self.parameters.copy()
        return new_model

    def __repr__(self):
        return f"ModelManager(name='{self.name}', expression='{self.expression}', parameters={self.parameters})"


class CommonModels:
    """
    Collection of commonly used X-ray spectral models.

    Note
    ----
    nH (neutral hydrogen column density) must be provided.
    Calculate using: `nh <RA> <Dec>` (HEASOFT command)
    """

    @staticmethod
    def absorbed_powerlaw(
        nH: float,
        PhoIndex: float = 1.0,
        freeze_nH: bool = True,
    ) -> ModelManager:
        """Absorbed power-law model: tbabs*powerlaw"""
        model = ModelManager("tbabs*powerlaw", name="absorbed_powerlaw")
        model.set_parameter("TBabs.nH", nH, frozen=freeze_nH)
        model.set_parameter("powerlaw.PhoIndex", PhoIndex, frozen=False)
        return model

    @staticmethod
    def absorbed_broken_powerlaw(
        nH: float,
        PhoIndex1: float = 1.0,
        PhoIndex2: float = 1.0,
        freeze_nH: bool = True,
    ) -> ModelManager:
        """Absorbed broken power-law model: tbabs*bknpower"""
        model = ModelManager("tbabs*bknpower", name="absorbed_bknpower")
        model.set_parameter("TBabs.nH", nH, frozen=freeze_nH)
        model.set_parameter("bknpower.PhoIndx1", PhoIndex1, frozen=False)
        model.set_parameter("bknpower.BreakE", 1.0, frozen=False)
        model.set_parameter("bknpower.PhoIndx2", PhoIndex2, frozen=False)
        return model

    @staticmethod
    def absorbed_logparabola(
        nH: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        freeze_nH: bool = True,
    ) -> ModelManager:
        """Absorbed log-parabola model: tbabs*logpar"""
        model = ModelManager("tbabs*logpar", name="absorbed_logpar")
        model.set_parameter("TBabs.nH", nH, frozen=freeze_nH)
        model.set_parameter("logpar.alpha", alpha, frozen=False)
        model.set_parameter("logpar.beta", beta, frozen=False)
        return model
