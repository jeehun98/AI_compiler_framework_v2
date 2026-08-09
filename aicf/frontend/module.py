class Module:
    """Base user-facing model object.

    TODO: parameter registration, buffers, training/eval state, nested modules.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
