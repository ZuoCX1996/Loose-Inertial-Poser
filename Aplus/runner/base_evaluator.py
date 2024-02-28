from torch.utils.data import DataLoader
import torch
from Aplus.utils import DataMeter

class BaseEvaluator:
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, device=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        avg_meter_loss = DataMeter()
        avg_meter_loss.reset()
        self.model.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)

            avg_meter_loss.update(value=loss.item(), n_sample=len(y))

        loss_train = avg_meter_loss.get_avg()
        return loss_train

    @classmethod
    def from_trainner(cls, trainner, data_eval):
        return cls(model=trainner.model, loss_func=trainner.loss_func, batch_size=trainner.batch_size, data=data_eval)
