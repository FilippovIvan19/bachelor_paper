import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from src.utils import calculate_metrics


class LitModule(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.history = {}
        self.optimizer = optimizer

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        self.log_dict(dic)
        return dic

    def validation_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        val_dic = {"val_" + k: v for k, v in dic.items()}
        self.log_dict(val_dic)
        return val_dic

    def test_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        test_dic = {"test_" + k: v for k, v in dic.items()}
        self.log_dict(test_dic)
        return test_dic

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        precision, accuracy, recall = calculate_metrics(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        # metrics = {"loss": loss, "precision": precision, "accuracy": accuracy, "recall": recall}
        metrics = {"loss": loss, "recall": recall}
        return metrics

    def make_epoch_metrics(self, step_outputs):
        metrics = {}
        for dic in step_outputs:
            for k, v in dic.items():
                metrics[k] = metrics.get(k, []) + [v]

        for k, v in metrics.items():
            metrics[k] = (sum(metrics[k]) / (len(metrics[k]))).item()

        for k, v in metrics.items():
            self.history[k] = self.history.get(k, []) + [v]
        return metrics

    def training_epoch_end(self, training_step_outputs):
        self.make_epoch_metrics(training_step_outputs)
        # self.history["epoch"] = self.history.get("epoch", []) + [self.trainer.current_epoch]

    def validation_epoch_end(self, validation_step_outputs):
        self.make_epoch_metrics(validation_step_outputs)

    def on_train_end(self):
        for k, v in self.history.items():
            if "val_" in k and len(v) > len(self.history.get(k[4:], [])):
                self.history[k] = self.history[k][1:]

    def configure_optimizers(self):
        return self.optimizer
