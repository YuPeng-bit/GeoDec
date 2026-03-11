import argparse
import contextlib
import logging
import os
import sys
# ==============================================================================
# [新增] 强制修复 torch.load 的 weights_only 问题
# ==============================================================================
import torch
from torch.serialization import add_safe_globals
import omegaconf

# 方法 A: 注册安全全局变量 (官方推荐，更优雅)
# 允许 OmegaConf 的配置对象被反序列化
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
except ImportError:
    pass

# 方法 B: 暴力热修补 (Aggressive Monkey Patch)
# 无论 PyTorch Lightning 传入什么，这里都强制改为 weights_only=False
_original_torch_load = torch.load

def aggressive_safe_load(*args, **kwargs):
    # 强制覆盖参数，即使调用者显式传了 True
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    # 如果没传，也强制设为 False (针对 PyTorch 2.6+ 的默认行为)
    else:
        kwargs['weights_only'] = False
        
    return _original_torch_load(*args, **kwargs)

torch.load = aggressive_safe_load
print("[INFO] Applied aggressive patch to torch.load (forcing weights_only=False)")
# ==============================================================================

class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pytorch_lightning as pl
    import torch
    import omegaconf
    # [新增] 修复 PyTorch 2.6+ 无法加载含 OmegaConf 的 Checkpoint 问题
    # 将 ListConfig 和 DictConfig 加入安全白名单
    try:
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig, 
            omegaconf.dictconfig.DictConfig
        ])
    except AttributeError:
        # 如果是旧版本 PyTorch，没有这个函数，直接跳过即可
        pass
    except Exception as e:
        print(f"Warning: Failed to add safe globals: {e}")


    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("mvadapter", "typeguard.typechecked")

    from mvadapter.systems.base import BaseSystem
    from mvadapter.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from mvadapter.utils.config import ExperimentConfig, load_config
    from mvadapter.utils.core import find
    from mvadapter.utils.misc import get_rank, time_recorder
    from mvadapter.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.benchmark:
        time_recorder.enable(True)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    dm = find(cfg.data_cls)(cfg.data)
    system: BaseSystem = find(cfg.system_cls)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
            CustomProgressBar(refresh_rate=1),
        ]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        ]
        if args.wandb:
            wandb_logger = WandbLogger(
                project="MV-Adapter", name=f"{cfg.name}-{cfg.tag}"
            )
            system._wandb_logger = wandb_logger
            loggers += [wandb_logger]
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    # set a different seed for each device
    # NOTE: use trainer.global_rank instead of get_rank() to avoid getting the local rank
    pl.seed_everything(cfg.seed + trainer.global_rank, workers=True)

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])
    if args.resume is not None:
        # 可选：如果需要在日志里记录一下
        print(f"Resuming from checkpoint: {args.resume}")
    if args.train:
        # 关键修改：加上 ckpt_path=args.resume
        trainer.fit(system, datamodule=dm, ckpt_path=args.resume)

    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume) # 验证时也可以加

    elif args.test:
        # 测试逻辑保持原样，或者也可以统一用 args.resume
        # 原代码好像用了 cfg.resume，你可以根据需要统一
        ckpt = args.resume if args.resume is not None else cfg.resume
        set_system_status(system, ckpt)
        trainer.test(system, datamodule=dm, ckpt_path=ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="path to checkpoint to resume training from"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")

    parser.add_argument("--wandb", action="store_true", help="if true, log to wandb")

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="if true, set to benchmark mode to record running times",
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()
    main(args, extras)
