
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import RoofMetrics, box_iou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.plotting import plot_images


class RoofValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a Roof model.

    This validator specializes in evaluating models that predict bounding boxes with rooftop
    direction attributes for building footprint detection in aerial imagery.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (RoofMetrics): Metrics object for evaluating Roof model performance.
        is_dota (bool): Flag indicating whether the validation dataset is in DOTA format.

    Methods:
        init_metrics: Initialize evaluation metrics for YOLO.
        postprocess: Extract rooftop directions from predictions.
        _prepare_batch: Prepare batch data including rooftop ground truth.
        _process_batch: Process batch of detections and compute rooftop metrics.
        get_desc: Return formatted string for metrics display.
        plot_predictions: Plot predicted bounding boxes on input images.
        pred_to_json: Serialize YOLO predictions to COCO json format.
        scale_preds: Scale predictions to original image size.
        eval_json: Evaluate YOLO output in JSON format.

    Examples:
        >>> from ultralytics.models.yolo.roof import RoofValidator
        >>> args = dict(model="yolo11n-roof.pt", data="roof-dataset.yaml")
        >>> validator = RoofValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize RoofValidator and set task to 'roof', metrics to RoofMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict | SimpleNamespace, optional): Arguments containing validation parameters.
            _callbacks (list, optional): List of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "roof"
        self.metrics = RoofMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO roof validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format
        self.confusion_matrix.task = "roof"  # set confusion matrix task to 'roof'

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Extract rooftop directions from 'extra' field.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions with rooftop directions.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            # Extract first 5 values as rooftop directions (logits, not sigmoid)
            if "extra" in pred and pred["extra"].shape[0] > 0:
                pred["rooftop"] = pred["extra"][:, :5]
            else:
                pred["rooftop"] = torch.zeros((0, 5), device=pred["bboxes"].device)
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare batch including rooftop ground truth.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations including rooftop data.
        """
        pbatch = super()._prepare_batch(si, batch)

        # Add rooftop ground truth if available
        if "extras" in batch:
            idx = batch["batch_idx"] == si
            if idx.sum() > 0:
                rooftop_gt = batch["extras"][idx][:, :5]  # First 5 are rooftop directions
                pbatch["rooftop"] = rooftop_gt
            else:
                pbatch["rooftop"] = torch.zeros((0, 5), device=self.device)
        else:
            pbatch["rooftop"] = torch.zeros((0, 5), device=self.device)

        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Process batch including rooftop accuracy computation.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data.
            batch (dict[str, Any]): Dictionary containing ground truth data.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing detection and rooftop metrics.
        """
        # Get standard detection metrics (tp for bounding boxes)
        tp = super()._process_batch(preds, batch)

        # Initialize rooftop metrics
        n_preds = preds["cls"].shape[0]
        tp_roof = np.zeros((n_preds, 5), dtype=bool)  # 5 directions

        # Compute rooftop metrics if both predictions and ground truth exist
        if (batch["cls"].shape[0] > 0 and n_preds > 0 and
                "rooftop" in preds and "rooftop" in batch and
                preds["rooftop"].shape[0] > 0 and batch["rooftop"].shape[0] > 0):

            # Match predictions to ground truth based on bbox IoU
            iou = box_iou(batch["bboxes"].cpu(), preds["bboxes"].cpu())  # (n_gt, n_pred)

            # For each prediction, find best matching ground truth
            if iou.numel() > 0:
                max_iou, matched_idx = iou.max(dim=0)  # (n_pred,)

                # Only compute rooftop metrics for good matches (IoU > 0.5)
                valid_matches = max_iou > 0.3

                if valid_matches.sum() > 0:
                    # Get matched ground truth rooftops
                    matched_gt_rooftop = batch["rooftop"][matched_idx]  # (n_pred, 5)

                    # Apply sigmoid to predictions and threshold at 0.5
                    pred_rooftop_probs = preds["rooftop"].sigmoid()
                    pred_rooftop_binary = (pred_rooftop_probs > 0.5)

                    # Ground truth should already be binary (0 or 1)
                    gt_rooftop_binary = matched_gt_rooftop > 0.1

                    # Compute per-direction correctness (only for valid matches)
                    tp_roof_valid = (pred_rooftop_binary.cpu() == gt_rooftop_binary.cpu()).cpu().numpy()

                    # Only count as correct if the bbox match is valid
                    valid_matches_np = valid_matches.cpu().numpy()
                    tp_roof = tp_roof_valid & valid_matches_np[:, None]

        tp["tp_roof"] = tp_roof  # Add rooftop true positives
        return tp

    def get_desc(self) -> str:
        """Return description including rooftop metrics."""
        return ("%22s" + "%11s" * 12) % (  # <-- 12 columns after "Class"
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Roof(Acc",
            "N",
            "S",
            "E",
            "W",
            "O)",
        )