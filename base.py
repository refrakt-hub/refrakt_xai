class BaseXAI:
    def __init__(self, model, **kwargs):
        self.model = model
        self.params = kwargs

    def explain(self, input_tensor, target=None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.") 