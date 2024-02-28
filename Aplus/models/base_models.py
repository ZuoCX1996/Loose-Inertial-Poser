import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def capture_output(self, x:torch.Tensor, module_name:str) -> torch.Tensor:
        """
        Capture specific module's output in the model.
        If module named [module_name] is not found in the model, return [None]
        :param x: Model input
        :param module_name: Name of module (e.g. 'fc_1')
        :return: Output of module named [module_name].

        """

        self.eval()
        module_output = []

        def hook_out_feat_record(module, input, output):
            module_output.append(output)
            return None

        hook_handles = []
        for name, moudle in self.named_children():
            if name == module_name:
                print(f'Target Module :({name}) {moudle}')
                h = moudle.register_forward_hook(hook=hook_out_feat_record)
                hook_handles.append(h)

        if len(hook_handles) == 0:
            print(f'module [{module_name}] doesn\'t exist')
            return None

        _ = self.forward(x)

        for h in hook_handles:
            h.remove()

        return module_output[0]

    def capture_input(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """
        Capture specific module's input in the model.
        If module named [module_name] is not found in the model, return [None]
        :param x: Model input
        :param module_name: Name of module (e.g. 'fc_1')
        :return: Input of module named [module_name].

        """
        self.eval()
        module_input = []

        def hook_in_feat_record(module, input, output):
            module_input.append(input)
            return None

        hook_handles = []
        for name, moudle in self.named_children():
            if name == module_name:
                print(f'Target Module :({name}) {moudle}')
                h = moudle.register_forward_hook(hook=hook_in_feat_record)
                hook_handles.append(h)

        if len(hook_handles) == 0:
            print(f'module [{module_name}] doesn\'t exist')
            return None

        _ = self.forward(x)

        for h in hook_handles:
            h.remove()

        return module_input[0][0]


    def save(self, path='./checkpoint.pth'):
        save_dict = {'model': self.state_dict()}
        torch.save(save_dict, path)

    def restore(self, checkpoint_path: str):
        try:
            self.load_state_dict(torch.load(checkpoint_path)['model'])
            print(f"Restore from {checkpoint_path}")
        except FileNotFoundError as r:
            print(f"File doesn't exist: {checkpoint_path}")

    def export_onnx(self, input_shapes:dict, output_shapes:dict, path):
        """
        Save model to onnx.
        When setting input/output shape, use '-1' to represent dynamic axes (e.g. the axes of [batch]).
        :param input_shapes: Shape of input tensor.
        :param output_shapes: Shape of output tensor.
        :param path: Path for model export, should end with '.onnx'.
        :return: None
        """
        self.eval()
        current_device = next(self.parameters()).device
        dummy_input_list = []
        dynamic_axes = {}
        input_names = []
        output_names = []

        for name, shape in input_shapes.items():

            input_names.append(name)
            _dynamic_axes = {}

            for i, v in enumerate(shape):
                if v == -1:
                    shape[i] = 1
                    _dynamic_axes.update({i:f'dim_{i}'})

            dummy_input_list.append(torch.rand(size=shape).to(current_device))

            if len(_dynamic_axes) > 0:
                dynamic_axes.update({name:_dynamic_axes})

        for name, shape in output_shapes.items():

            output_names.append(name)
            _dynamic_axes = {}

            for i, v in enumerate(shape):
                if v == -1:
                    shape[i] = 1
                    _dynamic_axes.update({i: f'dim_{i}'})

            if len(_dynamic_axes) > 0:
                dynamic_axes.update({name: _dynamic_axes})

        if len(dummy_input_list) > 0:
            dummy_input = tuple(dummy_input_list)
        else:
            dummy_input = dummy_input_list[0]


        torch.onnx.export(self, dummy_input, path, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes, opset_version=14)


