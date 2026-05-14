import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(
        self,
        model,
        discriminator,
        criterion_g,
        criterion_d,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g=None,
        lr_scheduler_d=None,
        *args,
        **kwargs
    ):
        """
        Args:
            model: generator network.
            discriminator: discriminator network.
            criterion_g: loss function for generator.
            criterion_d: loss function for discriminator.
            optimizer_g: optimizer for generator.
            optimizer_d: optimizer for discriminator.
            lr_scheduler_g: learning rate for generator.
            lr_sceduler_d: learning rate for discriminator.
        """
        super().__init__(
            model=model,
            criterion=criterion_g,
            optimizer=optimizer_g,
            lr_scheduler=lr_scheduler_g,
            *args,
            **kwargs
        )
        self.discriminator = discriminator
        self.criterion_d = criterion_d
        self.optimizer_d = optimizer_d
        self.lr_scheduler_d = lr_scheduler_d

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        x_real = batch["audio"]

        outputs = self.model(x_real)
        x_fake = outputs["fake_audio"]
        batch.update(outputs)

        if self.is_train:
            # train discriminator
            self.optimizer_d.zero_grad()

            d_real_scores, _ = self.discriminator(x_real)
            d_fake_scores_det, _ = self.discriminator(x_fake.detach())

            loss_d = self.criterion_d(d_real_scores, d_fake_scores_det)["loss"]

            loss_d.backward()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            batch["loss_d"] = loss_d

            # train generator
            self.optimizer.zero_grad()

            d_fake_scores, d_fake_features = self.discriminator(x_fake)
            _, d_real_features = self.discriminator(x_real)

            g_loss_dict = self.criterion(
                x_real=x_real,
                x_fake=x_fake,
                d_real_features=d_real_features,
                d_fake_features=d_fake_features,
                d_fake_scores=d_fake_scores,
            )

            loss_g = g_loss_dict["loss"] + outputs["commitment_loss"]

            loss_g.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            batch["loss_g"] = loss_g
            batch.update(
                {key: value for key, value in g_loss_dict.items() if key != "loss"}
            )
        else:
            d_fake_scores, d_fake_features = self.discriminator(x_fake)
            _, d_real_features = self.discriminator(x_real)

            g_loss_dict = self.criterion(
                x_real=x_real,
                x_fake=x_fake,
                d_real_features=d_real_features,
                d_fake_features=d_fake_features,
                d_fake_scores=d_fake_scores,
            )

            loss_g = g_loss_dict["loss"] + outputs["commitment_loss"]
            batch["loss_g"] = loss_g
            batch.update(
                {key: value for key, value in g_loss_dict.items() if key != "loss"}
            )

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        batch["loss"] = batch["loss_g"]

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            if batch_idx % self.log_step == 0:
                self.writer.add_audio(
                    "original_audio", batch["audio"][0], sample_rate=16000
                )
                self.writer.add_audio(
                    "reconstructed_audio", batch["fake_audio"][0], sample_rate=16000
                )
        else:
            if batch_idx % self.log_step == 0:
                self.writer.add_audio(
                    "eval_original_audio", batch["audio"][0], sample_rate=16000
                )
                self.writer.add_audio(
                    "eval_reconstructed_audio",
                    batch["fake_audio"][0],
                    sample_rate=16000,
                )
