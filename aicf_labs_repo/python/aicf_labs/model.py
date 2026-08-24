"""Base model declaration and minimal eager execution API."""


class Model:
    """A model whose explicit operator calls can be eagerly run or traced."""

    def forward(self, *args: object, **kwargs: object) -> object:
        """Execute the model; concrete executable models override this method."""

        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)

    def summary(self) -> str:
        """Return a human-readable structural summary."""

        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()
