"Part of the code has been taken from"

"4DHumans: https://github.com/shubham-goel/4D-Humans"

from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.camerahmr_trainer import CameraHMR
from core.utils.pylogger import get_pylogger
from core.utils.misc import task_wrapper, log_hyperparameters
from pytorch_lightning.strategies import DDPStrategy
import signal

signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch

torch.set_float32_matmul_precision("medium")
torch.manual_seed(0)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, "model_config.yaml"))
    with open(os.path.join(rootdir, "dataset_config.yaml"), "w") as f:
        f.write(dataset_cfg.dump())


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = DataModule(cfg, dataset_cfg)

    # Setup model
    model = CameraHMR(cfg)

    logger = TensorBoardLogger(
        os.path.join(cfg.paths.output_dir, "tensorboard"),
        name="",
        version="",
        default_hp_metric=False,
    )
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor="val_loss",
    )
    rich_callback = pl.callbacks.TQDMProgressBar(refresh_rate=100)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor, rich_callback]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=500,
        val_check_interval=cfg.trainer.val_check_interval,
        precision="16-mixed",
        max_steps=cfg.trainer.max_steps,
        logger=loggers,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    trainer.fit(model, datamodule=datamodule, ckpt_path="last")

    log.info("Fitting done")


@hydra.main(
    version_base="1.2",
    config_path=str(root / "core/configs_hydra"),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
