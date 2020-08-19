
class Captain:
    def __init__(self):
        self.data = {}

    def hook(self, name):
        def hook_fn(model, input, output):
            self.data[name] = output.detach()
        return hook_fn