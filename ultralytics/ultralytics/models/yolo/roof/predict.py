import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class RoofPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Roof model.

    This predictor handles identification of roof detection tasks, processing images and returning results with roof topology
    characteristics.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO Roof model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.roof import RoofPredictor
        >>> args = dict(model="yolo11n-roof.pt", source=ASSETS)
        >>> predictor = RoofPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize RoofPredictor with optional model and data configuration overrides.

        Args:
            cfg (dict, optional): Default configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over the default config.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.roof import RoofPredictor
            >>> args = dict(model="yolo11n-roof.pt", source=ASSETS)
            >>> predictor = RoofPredictor(overrides=args)
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "roof"

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rooftop properties with shape (N, 18) where
                the last dimension contains [x, y, w, h, confidence, class_id, var1,var2,var3]. 6+12
            img (torch.Tensor): The image after preprocessing with shape (B, C, H, W).
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and oriented bounding
                boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape, xywh=True)
        roofs = pred[:, 6:]
        return Results(orig_img, path=img_path, names=self.model.names, boxes = pred[:,:6], roof=roofs)