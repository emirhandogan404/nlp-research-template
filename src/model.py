import lightning as L
from print_on_steroids import logger
import torch
from torch.optim import Adam

from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


class EfficientNetV2Wrapper(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        lr_schedule: str,
        warmup_epochs: int,
        lr_decay: float,
        lr_decay_interval: int,
        beta1: float,
        beta2: float,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        margin: float = 0.5,
    ) -> None:
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Remove LR scheduling for now
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.margin = margin

        ##### Load the model here #####
        if from_scratch:
            logger.info("Loading model from scratch")
            self.model = efficientnet_v2_m()
        else:
            logger.info(f"Loading model {model_name_or_path}")
            self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            # self.model = efficientnet_v2_m(weights=None)

        # remove the last fully connected layer
        # self.model.classifier = torch.nn.Identity()
        self.model.classifier = torch.nn.Linear(1280, 256)  # TODO: parameterize this via args / config
        ###############################

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch):
        anchor, positive, negative = batch

        # feed the anchor, positive, negative images to the model
        anchor = self(anchor)
        positive = self(positive)
        negative = self(negative)

        loss, distance_positive, distance_negative = self.triplet_loss(anchor, positive, negative, margin=self.margin)
        return loss, distance_positive, distance_negative

        # return self.triplet_loss(anchor, positive, negative)

    def triplet_loss(self, anchor, positive, negative, margin=0.5):
        distance_positive = torch.functional.norm(anchor - positive, dim=1)
        distance_negative = torch.functional.norm(anchor - negative, dim=1)
        losses = torch.relu(distance_positive - distance_negative + margin).mean()
        return losses.mean(), distance_positive.mean(), distance_negative.mean()

    def training_step(self, batch, batch_idx):
        loss, distance_positive, distance_negative = self._calculate_loss(batch=batch)
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_distance_positive", distance_positive, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_distance_negative", distance_negative, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, distance_positive, distance_negative = self._calculate_loss(batch=batch)
        self.log("val_loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        self.log("val_distance_positive", distance_positive, on_step=True, prog_bar=True, sync_dist=True)
        self.log("val_distance_negative", distance_negative, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup epochs: {self.warmup_epochs}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adam(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs * self.learning_rate
            else:
                num_decay_cycles = (epoch - self.warmup_epochs) // self.lr_decay_interval
                return (self.lr_decay**num_decay_cycles) * self.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_decay_interval},
        }