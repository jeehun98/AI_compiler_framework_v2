from .executable import Executable


class Executor:
    def execute(self, executable: Executable, *args, **kwargs):
        return executable.run(*args, **kwargs)
