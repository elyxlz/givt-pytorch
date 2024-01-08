from dataclasses import dataclass, asdict
import os
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel, get_inverse_sqrt_schedule
from ema_pytorch import EMA
import wandb

from .model import GIVT
from .data import DummyDataset


@dataclass
class TrainConfig:
    # main
    name: str = "givt-test"
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    ema: float | None = None
    cpu: bool = False

    # dataset
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # optimizer
    lr: float = 1e-4
    weight_decay: float = 0.1
    lr_eps: float = 1e-8
    lr_betas: tuple = (0.9, 0.99)
    fused_adam: bool = False
    grad_norm: float = 1.0

    # scheduler
    warmup_steps: int = 1000
    timescale: int = 1e8
    last_epoch: int = -1

    # logging and checkpointing
    log_every: int = 100
    save_every: int = 1000
    push_every: int | None = None
    val_every: int | None = None
    resume_from_ckpt: str | None = None
    use_wandb: bool = False
    wandb_project_name: str = "givt-pytorch"

    def to_dict(self):
        return asdict(self)


class Trainer:
    def __init__(
        self,
        model: GIVT,
        dataset: DummyDataset,
        train_config: TrainConfig,
    ):
        self.train_config = train_config

        self.completed_steps = -1
        self.run_id = wandb.util.generate_id()

        # ckpt
        if train_config.resume_from_ckpt is not None:
            self.run_id = os.path.basename(
                os.path.dirname(train_config.resume_from_ckpt)
            )

        # log dir
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.log_dir = os.path.join(os.path.join(root_dir, "logs"), self.run_id)

        self.accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            cpu=train_config.cpu,
        )

        # wandb
        if train_config.use_wandb and self.accelerator.is_local_main_process:
            config = train_config.to_dict() | model.config.to_dict()
            config["seed"] = os.getenv("GLOBAL_SEED")
            assert (
                train_config.wandb_project_name is not None
            ), "Please provide a wandb project name"
            self.accelerator.init_trackers(
                project_name=train_config.wandb_project_name,
                config=config,
                init_kwargs=dict(
                    wandb=dict(
                        name=train_config.name,
                        dir=os.path.join(root_dir, "logs"),
                        id=self.run_id,
                        resume="allow",
                    )
                ),
            )

        # model
        self.model: PreTrainedModel = self.accelerator.prepare(model)
        self.model.train()

        # ema
        self.model_ema = None
        if train_config.ema is not None:
            self.model_ema = EMA(
                self.model,
                power=train_config.ema,
                karras_beta=True,
                include_online_model=False,
            )
            self.accelerator.register_for_checkpointing(self.model_ema.ema_model)

        if train_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # optimizer
        nodecay_params = [p for p in self.model.parameters() if p.dim() == 1]
        decay_params = [p for p in self.model.parameters() if p.dim() != 1]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": nodecay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": train_config.weight_decay},
            ],
            lr=train_config.lr,
            betas=train_config.lr_betas,
            eps=train_config.lr_eps,
            fused=train_config.fused_adam,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # scheduler
        self.scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            num_warmup_steps=train_config.warmup_steps,
            timescale=train_config.timescale,
            last_epoch=train_config.last_epoch,
        )
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.scheduler = self.accelerator.prepare(self.scheduler)

        # dataloaders
        self.train_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            shuffle=False,
        )

        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )

        if train_config.resume_from_ckpt is not None:
            self.resume()

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.accelerator.print("Trainable parameters: ", trainable_params / 1e6, "M")

    def training_step(self, batch):
        out = self.model.forward(**batch)
        loss = out.loss
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.train_config.grad_norm
        )  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        if self.time_to_log():
            self.accelerator.log(out.info, step=self.completed_steps)
            self.accelerator.log({"train_loss": loss}, step=self.completed_steps)
            self.epoch_bar.set_postfix({"loss": loss.item()})

    @torch.no_grad()
    def validation(self, val_loader):
        """Validation loop"""
        model = self.model_ema if self.model_ema is not None else self.model
        model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validation"):
            loss = model.forward(**batch).loss
            val_loss += loss.item() / len(val_loader)
        self.accelerator.log({"val_loss": loss})
        model.train()

    def train(self):
        """Basic training loop"""

        epochs = -1
        while True:
            epochs += 1

            self.accelerator.log({"epoch": epochs}, step=self.completed_steps)

            self.epoch_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epochs}",
                disable=not self.accelerator.is_local_main_process,
                initial=self.completed_steps + 1 % len(self.train_loader),
            )

            for batch in self.epoch_bar:
                self.completed_steps += 1

                with self.accelerator.accumulate(self.model):
                    self.training_step(batch)

                # validation and evaluation
                if self.time_to_val():
                    self.accelerator.print(f"Validation at step {self.completed_steps}")
                    self.validation(self.val_loader)

                # checkpoint
                if self.time_to_save():
                    self.save()

                # push to hub
                if self.time_to_push():
                    model = self.model_ema if self.model_ema is not None else self.model
                    self.push(model)

    def time_to_save(self) -> bool:
        save: bool = (
            self.train_config.save_every is not None
            and self.completed_steps % self.train_config.save_every == 0
            and self.completed_steps != 0
        )
        return save

    def time_to_push(self) -> bool:
        push: bool = (
            self.train_config.push_every is not None
            and self.completed_steps % self.train_config.push_every == 0
            and self.completed_steps != 0
        )
        return push

    def time_to_log(self) -> bool:
        log: bool = (
            self.train_config.log_every is not None
            and self.completed_steps % self.train_config.log_every == 0
        )
        return log

    def time_to_val(self) -> bool:
        val: bool = (
            self.train_config.val_every is not None
            and self.completed_steps % self.train_config.val_every == 0
        )
        return val

    def save(self) -> None:
        """Saves model to path"""

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.accelerator.is_local_main_process:
            self.accelerator.print(f"Saving model to {self.log_dir}")
            save_dir = os.path.join(self.log_dir, f"step_{self.completed_steps}")
            # check if there are < 5 checkpoints
            if len(os.listdir(self.log_dir)) > 5:
                # remove oldest checkpoint
                oldest_step = sorted(
                    [
                        int(os.path.splitext(f)[0].replace("step_", ""))
                        for f in os.listdir(self.log_dir)
                    ]
                )[0]
                oldest_step_dir = os.path.join(self.log_dir, f"step_{oldest_step}")
                self.accelerator.print(f"Removing oldest checkpoint {oldest_step_dir}")
                os.system(f"rm -rf {oldest_step_dir}")
            self.accelerator.save_state(save_dir)

    def resume(self) -> None:
        """Resumes from checkpoint state, sets self.resume_step and adds to self.completed_steps"""
        assert (
            self.train_config.resume_from_ckpt is not None
        ), "Please provide a checkpoint path to resume from"
        self.accelerator.print(
            f"Resuming from checkpoint {self.train_config.resume_from_ckpt}"
        )
        self.accelerator.load_state(self.train_config.resume_from_ckpt)
        path = os.path.basename(self.train_config.resume_from_ckpt)
        training_basename = os.path.splitext(path)[0]
        resume_step = int(training_basename.replace("step_", ""))
        self.completed_steps += resume_step

    def push(self, model: PreTrainedModel) -> None:
        """Takes care of pushing the model to hub, multi-rank safe"""
        try:
            if self.completed_steps == 0:
                return

            if self.accelerator.is_main_process:
                self.accelerator.print("Pushing model to hub...")
                unwrapped_model: GIVT = self.accelerator.unwrap_model(model)

                unwrapped_model.push_to_hub(
                    self.train_config.hub_namespace,
                    commit_message=f"Run {self.run_id}, step {self.completed_steps}",
                    private=True,
                    token=os.environ["HUGGINGFACE_TOKEN"],
                )
            self.accelerator.wait_for_everyone()
        except Exception as e:
            print("Push failed", e)
            self.accelerator.wait_for_everyone()
