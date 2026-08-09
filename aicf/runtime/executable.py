from dataclasses import dataclass


@dataclass
class Executable:
    image: object

    def run(self, *args, **kwargs):
        # TODO: bind buffers and launch actual kernels.
        return {
            "status": "mock",
            "kernels": list(getattr(self.image, "kernels", [])),
        }
