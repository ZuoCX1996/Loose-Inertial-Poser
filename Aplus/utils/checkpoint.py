import torch
class CheckPoint():
    def __init__(self, model, optimizer, log_manager):
        self.model = model
        self.optimizer = optimizer
        self.log_manager = log_manager

    def save(self, save_folder_path, epoch, model_name=None):
        save_state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            'log': self.log_manager.log,
        }
        # 多个optimizer可以以list存储
        if isinstance(self.optimizer, list):
            optimizer_states = []
            for i, optim in enumerate(self.optimizer):
                optimizer_states.append(optim.state_dict())
            save_state.update({'optimizer': optimizer_states})
        else:
            save_state.update({'optimizer': self.optimizer.state_dict()})
        if model_name is None:
            model_name = type(self.model).__name__
        torch.save(save_state, f'{save_folder_path}/{model_name}_{epoch}.pth')
    @staticmethod
    def load(file_path):
        try:
            checkpoint_dict = torch.load(file_path)
            print(f"check point [{file_path}] loaded")
        except FileNotFoundError as r:
            print(f"Error: check point [{file_path}] doesn't exist")

        return checkpoint_dict



