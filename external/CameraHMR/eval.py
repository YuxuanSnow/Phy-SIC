from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import warnings

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from core.configs import dataset_config
from core.constants import CAM_MODEL_CKPT, CHECKPOINT_PATH
from core.datasets import DataModule
from core.camerahmr_trainer import CameraHMR

from core.utils.pylogger import get_pylogger
import signal

signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch

torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)
warnings.filterwarnings("ignore", category=UserWarning)


def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # Load dataset config
    dataset_cfg = dataset_config()

    datamodule = DataModule(cfg, dataset_cfg)
    model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    cam_model_checkpoint = torch.load(CAM_MODEL_CKPT)["state_dict"]
    model.cam_model.load_state_dict(cam_model_checkpoint)

    trainer = pl.Trainer()

    trainer.test(model, datamodule=datamodule)
    log.info("Fitting done")


@hydra.main(
    version_base="1.2",
    config_path=str(root / "core/configs_hydra"),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    eval(cfg)


if __name__ == "__main__":
    main()
