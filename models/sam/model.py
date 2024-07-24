from typing import List, Dict, Optional, Tuple

import logging
import pathlib
import os
import requests
from uuid import uuid4

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import InMemoryLRUDictCache
from label_studio_converter import brush

import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.amg import build_all_layer_point_grids

from ..mediaret import MediaCache
from ..utils import masks2polypoints, pix2pc, simplify_polygon

logger = logging.getLogger("SamModel")


def get_model(model_type: str, output_path: os.PathLike):
    """Download models"""

    if model_type == "vit_h":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    elif model_type == "vit_l":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    elif model_type == "vit_b":
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    else:
        raise ValueError(f"Unknown SAM model type {model_type}")

    file_name = url.split("/")[-1]

    output_path = pathlib.Path(output_path) / file_name
    if not output_path.exists():
        req = requests.get(url)
        with open(output_path, "wb") as f:
            for chunk in req.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    return {
        "model_path": output_path,
    }


class _InternalSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):

    def __init__(
        self,
        predictor: SamPredictor,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        This is copied from the SAM model, we want to provide a pre-initiated predictor
        to avoid loading the model twice.

        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (SamPredictor): The initiated predictor
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = predictor
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode


class SamModel(LabelStudioMLBase, SamAutomaticMaskGenerator):
    """Custom ML Backend model for Segment Anything Model

    [Link]

    Env Vars:
        LSBE_MODEL_PATH
        LSBE_SAM_MODEL_TYPE [vit_l, vit_b, vit_h]
    """

    def __init__(
        self,
        model_name: str = "SamModel",
        project_id: Optional[str] = None,
        label_config=None,
    ):
        super().__init__(project_id=project_id, label_config=label_config)
        self.model_name = model_name
        self.cache = MediaCache(10)

        if path := os.environ.get("LSBE_MODEL_PATH"):
            self.model_ckpt_path = pathlib.Path(path)
        else:
            raise ValueError("LSBE_MODEL_PATH not set")
        self.model_ckpt_type = os.environ.get("LSBE_SAM_MODEL_TYPE")
        if not self.model_ckpt_path.exists():
            raise FileNotFoundError(
                f"The SAM checkpoint was not found for LSBE_MODEL_PATH={self.model_ckpt_path}"
            )
        logger.info(
            f"Using SAM model {self.model_ckpt_path} with type {self.model_ckpt_type}"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device {self.device}")

        ckpoint = sam_model_registry[self.model_ckpt_type](
            checkpoint=self.model_ckpt_path
        )
        logger.debug(f"added model to registry")
        ckpoint.to(device=self.device)
        self.predictor = SamPredictor(ckpoint)
        logger.debug(f"Model Loaded")

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.1.0")

    def _get_points(self, context: Dict) -> Tuple[List, List]:
        """Process the context to return SAM compatible points"""
        keypoints = []
        keypoint_labels = []
        for res in filter(lambda x: x["type"] == "keypointlabels", context["result"]):
            pix_x = int(res["value"]["x"] * res["original_width"] / 100)
            pix_y = int(res["value"]["y"] * res["original_height"] / 100)
            keypoints.append([pix_x, pix_y])
            keypoint_labels.append(int(res.get("is_positive", 1)))
        return (
            np.array(keypoints, dtype=np.float32) if keypoints else None,
            np.array(keypoint_labels, dtype=np.float32) if keypoint_labels else None,
        )

    def _get_rectangles(self, context: Dict) -> List:
        """Process the context to return SAM compatible boxes"""
        boxes = []
        for res in filter(lambda x: x["type"] == "rectanglelabels", context["result"]):
            if res["value"]["rotation"] != 0:
                logger.warn("Bounding boxes with rotation unsupported")
                continue
            else:
                pix_x = int(res["value"]["x"] * res["original_width"] / 100)
                pix_y = int(res["value"]["y"] * res["original_height"] / 100)
                box_width = res["value"]["width"] * res["original_width"] / 100
                box_height = res["value"]["height"] * res["original_height"] / 100
                boxes.append(
                    [pix_x, pix_y, int(box_width + pix_x), int(box_height + pix_y)]
                )

        if len(boxes) >= 1:
            logger.warn("Multiple bounding boxes unsupported - using first box only")
        return np.array(boxes[0]) if len(boxes) == 1 else None

    def set_image(self, img: str, calculate_embeddings=True, task=None):
        """Get the image and add it to the model.

        Args:
            img:
            calculate_embeddings
            task

        Returns:
            payload from cache or calculated payload
        """
        cache_payload, img_path = self.cache.get_media(img, task)
        if cache_payload is None:
            # the image was not in the cache
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            payload = {"image_shape": image.shape[:2]}
            if calculate_embeddings:
                image_embedding = self.predictor.get_image_embedding().cpu().numpy()
                payload["image_embedding"] = image_embedding
            self.cache.cache_media(img, payload)
            return payload
        else:
            return cache_payload

    def get_predict_extra_params(self) -> Dict:
        """Get the extra parameters for controlling returned output.

        Extra Params:
            polypoint_tolerance: float
            filter: ["largest", "none", ]
            return_masks: true
            return_polygons: true
        """
        ep = self.extra_params
        return {
            key: ep.get(key, default)
            for key, default in (
                ("polypoint_tolerance", 3.0),
                ("filter", "largest"),
                ("return_masks", True),
                ("return_polygons", True),
                ("auto_stability_score_thresh", 0.955),
                ("auto_pred_iou_thresh", 0.75),
                ("max_predictions", 25),
                ("min_mask_region_area", 25),
            )
        }

    def get_labels(self, result: Dict) -> List[str]:
        tag = result["type"]
        if tag in ["rectanglelabels", "keypointlabels"]:
            return result["value"][tag]
        else:
            return result["value"].get("labels", [])

    def _mask2results(self, masks, probs, width, height, labels):
        """Convert binary masks to rle brushes and polypoints"""
        extra_params = self.get_predict_extra_params()

        rles = [brush.mask2rle(mask * 255) for mask in masks]
        polypoints = masks2polypoints(masks, strategy="all")

        polypoints = [
            simplify_polygon(poly, tolerance=extra_params["polypoint_tolerance"])
            for poly in polypoints
        ]

        mask_results = [
            # mask result goes to brush
            {
                "id": str(uuid4())[:10],
                "from_name": "brush",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": labels,
                },
                "score": float(prb),
                "type": "brushlabels",
                "readonly": False,
                "origin": "prediction",
            }
            for prb, rle in zip(probs, rles)
        ]

        polygon_results = [
            {
                "id": str(uuid4())[:10],
                "from_name": "polygon",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "points": pix2pc(pp, width, height).tolist(),
                    "polygonlabels": labels,
                    "closed": True,
                },
                "type": "polygonlabels",
                "origin": "prediction",
            }
            for pp in polypoints
        ]
        return mask_results, polygon_results

    def _prompt_predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[dict]:
        """Interactive prediction processing. Takes point or box inputs and returns
        individual mask predictions from user prompts.

        """
        extra_params = self.get_predict_extra_params()

        # get keypoints
        keypoints, keypoint_labels = self._get_points(context)
        box = self._get_rectangles(context)

        # get the image data
        data = context["result"][0]
        width = data.get("original_width")
        height = data.get("original_height")
        labels = self.get_labels(data)
        img = tasks[0]["data"]["image"]

        self.set_image(img, task=tasks[0])

        masks, probs, logits = self.predictor.predict(
            point_coords=keypoints,
            point_labels=keypoint_labels,
            box=box,
            multimask_output=True,
        )

        if extra_params["filter"] == "largest":
            lg_index = masks.sum(axis=(1, 2)).argmax()
            masks = np.array([masks[lg_index, ...]])

        logger.debug(f"{len(probs)} mask suggestions")

        mask_results, polygon_results = self._mask2results(
            masks, probs, width, height, labels
        )

        predictions = [
            {
                "model_version": f"SAM {str(self.model_version)}",
                "score": float(np.average(probs)),
                "result": mask_results + polygon_results,
            }
        ]
        return predictions

    def _auto_predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """Automatic segmentation of tasks, this applies the default model and segments
        everything

        """
        extra_params = self.get_predict_extra_params()

        stability_score_thresh = extra_params.get("auto_stability_score_thresh")
        pred_iou_thresh = extra_params.get("auto_pred_iou_thresh")
        max_predictions = extra_params.get("max_predictions")
        min_mask_region_area = extra_params.get("min_mask_region_area")

        mask_generator = _InternalSamAutomaticMaskGenerator(
            self.predictor,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

        predictions = []

        for task in tasks:
            # get the image data
            img = task["data"]["image"]

            _, img_path = self.cache.get_media(img, task)
            # the image was not in the cache
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, _ = image.shape
            masks = mask_generator.generate(image)

            logger.debug(f"max_predictions: {max_predictions}")
            if max_predictions:
                # select the largest masks
                areas = [m["area"] for m in masks]
                areas_idx = np.argsort(areas)[-max_predictions:]
                logger.debug(f"areas_idx {areas_idx}")
                bin_masks = [masks[i]["segmentation"] for i in areas_idx]
                probs = [masks[i]["stability_score"] for i in areas_idx]
                logger.debug("len probs: {len(probs)}")
            else:
                bin_masks = [m["segmentation"] for m in masks]
                probs = [m["stability_score"] for m in masks]

            mask_results, polygon_results = self._mask2results(
                bin_masks, probs, width, height, []
            )

            predictions.append(
                {
                    "model_version": f"SAM {str(self.model_version)}",
                    "score": float(np.average(probs)),
                    "result": mask_results + polygon_results,
                }
            )

        return predictions

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        if context and context.get("result"):
            # the user has provided context, so we are in prompt based prediction
            logger.debug("Prompt Prediction")
            predictions = self._prompt_predict(tasks, context=context, **kwargs)
        elif tasks and context == None:
            logger.debug("Auto Segmentation")
            # apply auto segmentation to input tasks and return many predictions
            predictions = self._auto_predict(tasks, **kwargs)
        else:
            # if there is no context and no tasks return empty result
            logger.debug("No tasks to run")
            return []

        response = ModelResponse(predictions=predictions)
        # logger.debug(f"{response.model_dump()}")
        return response

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
