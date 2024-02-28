import pandas as pd
import torch


class LogManager:
    def __init__(self, items: list, log_data=None):
        """

        :param items: e.g. ['epoch', 'loss_train', 'loss_eval']
        :param log_data: Dict
        """
        if log_data is not None:
            self.log = log_data
        else:
            self.log = {}
            for item in items:
                self.log.update({item:[]})

    def update(self, values:dict):
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.cpu()
                if value.numel() == 1:
                    value = float(value)
            self.log[key].append(value)

    def print_latest(self):
        for key, value in self.log.items():
            last_value = self.log[key][-1]
            if type(last_value) == int or type(last_value) == float:
                last_value = round(last_value, 5)
            print(f'| {key}: {last_value } ', end='')
        print('|')

    def to_excel(self, path:str):
        df_log = pd.DataFrame.from_dict(self.log)
        df_log.to_excel(f'{path}', index=False)

    def load_data(self, data:dict):
        self.log = data

    @classmethod
    def from_dict(cls, data:dict):
        return cls(items=[], log_data=data)

    @classmethod
    def from_excel(cls, path):
        df = pd.read_excel(path)
        return cls(items=[], log_data=df.to_dict())

