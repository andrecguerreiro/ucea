
from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import RoofModel
from ultralytics.utils import DEFAULT_CFG, RANK


class RoofTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a Roof model.

    This trainer specializes in training YOLO models that detect roofs in images, while also estimating
    topology information

    Attributes:
        loss_names (tuple): Names of the loss components used during training including box_loss, cls_loss,
            and dfl_loss.

    Methods:
        get_model: Return RoofModel initialized with specified config and weights.
        get_validator: Return an instance of RoofValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.roof import RoofTrainer
        >>> args = dict(model="yolo11n-roof.pt", data="dota8.yaml", epochs=3)
        >>> trainer = RoofTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        """
        Initialize an RoofTrainer object for training Roof models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and
                model configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here
                will take precedence over those in cfg.
            _callbacks (list[Any], optional): List of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "roof"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> RoofModel:
        """
        Return RoofModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (RoofModel): Initialized RoofModel with the specified configuration and weights.

        Examples:
            >>> trainer = RoofTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-roof.yaml", weights="yolo11n-roof.pt")
        """
        model = RoofModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of RoofValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" , "clsrf_loss" , "area_loss" , "incli_loss" , "orien_loss" 
        return yolo.roof.RoofValidator( 
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )