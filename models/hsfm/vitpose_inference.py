import os
import torch
import numpy as np

from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from torch import nn
from typing import List, Dict, Tuple

os.environ["PYOPENGL_PLATFORM"] = "egl"

enable_offload = not bool(os.getenv("DISABLE_OFFLOAD"))


class ViTPoseModel:
    MODEL_DICT = {
        "ViTPose+-G (multi-task train, COCO)": {
            "config": "default_configs/thirdparty/vitpose/ViTPose_huge_wholebody_256x192.py",
            "model": "essentials/vitpose/ckpts/vitpose+_huge/wholebody.pth",
        },
    }

    def __init__(
        self,
        model_name: str = "ViTPose+-G (multi-task train, COCO)",
        model_config: str = None,
        model_checkpoint: str = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.device = torch.device(device)

        self.model_name = model_name
        self.MODEL_DICT[model_name] = {
            "config": model_config,
            "model": model_checkpoint,
        }

        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic["model"]
        model = init_pose_model(dic["config"], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(
            image, out, kpt_score_threshold, vis_dot_radius, vis_line_thickness
        )
        return out, vis

    def predict_pose(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float = 0.5,
    ) -> List[Dict[str, np.ndarray]]:
        """
        det_results: a list of Dict[str, np.ndarray] 'bbox': xyxyc
        """
        out, _ = inference_top_down_pose_model(
            self.model,
            image,
            person_results=det_results,
            bbox_thr=box_score_threshold,
            format="xyxy",
        )
        return out

    def visualize_pose_results(
        self,
        image: np.ndarray,
        pose_results: List[np.ndarray],
        kpt_score_threshold: float = 0.3,
        vis_dot_radius: int = 4,
        vis_line_thickness: int = 1,
    ) -> np.ndarray:
        vis = vis_pose_result(
            self.model,
            image,
            pose_results,
            kpt_score_thr=kpt_score_threshold,
            radius=vis_dot_radius,
            thickness=vis_line_thickness,
        )
        return vis


model = None


def load_vitpose():
    """
    Loads the VitPose model for 2D human pose estimation.
    This function initializes the model and moves it to the appropriate device (CPU or GPU).
    """
    global model
    if "model" not in globals() or model is None:
        model = ViTPoseModel(
            model_config="models/hsfm/configs/vitpose/ViTPose_huge_wholebody_256x192.py",
            model_checkpoint="data/vitpose_huge_wholebody.pth",
        )
        if enable_offload:
            model.model.cpu()
            torch.cuda.empty_cache()


def delete_vitpose():
    """
    Deletes the VitPose model from memory.
    This function is used to free up resources when the model is no longer needed.
    """
    global model
    if "model" in globals() and model is not None:
        del model
        model = None
        torch.cuda.empty_cache()


def get_human_pose_2d_vitpose(image_np: np.ndarray, boxes: List[np.ndarray]):
    """
    Calculates 2D human pose keypoints from an input image using the VitPose model.
    Parameters:
        image_np (np.ndarray): The input image in RGB format.
        boxes (List[np.ndarray]): A list of bounding box coordinates for detected humans.
            Each bounding box is expected in the format [x1, y1, x2, y2].
    Returns:
        tuple:
            - keypoints (np.ndarray): A numpy array of shape (N, num_keypoints, 2) containing the
              (x, y) coordinates of each detected keypoint for N humans.
            - scores (np.ndarray): A numpy array of shape (N, num_keypoints) containing the confidence
              scores for each corresponding keypoint.
    Note:
        - The function converts the image from RGB to BGR format as required by the VitPose model.
        - It temporarily moves the model to CUDA for inference, then returns it to the CPU and clears
          the CUDA cache.
    """
    if enable_offload:
        model.model.cuda()
    # VitPose takes BGR images. Reference: https://github.com/IRVLUTD/hamer-depth/blob/1f8ec0d475b208a39dd8e85fff3aeccd70cb9059/vitpose_model.py#L80
    image_cv = image_np[:, :, ::-1]
    boxes_in = []
    # convert to xyxyc format
    for i, box in enumerate(boxes):
        boxes_in.append({"bbox": np.concatenate([box, np.array([1])], axis=0)})
    results = model.predict_pose(image_cv, boxes_in, 0.0)
    result_keypoints = np.stack([result["keypoints"] for result in results])

    if enable_offload:
        model.model.cpu()
        torch.cuda.empty_cache()

    return result_keypoints[:, :, :2], result_keypoints[:, :, 2]
