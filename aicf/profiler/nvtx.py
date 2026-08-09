class NVTXRange:
    """Placeholder for optional NVTX instrumentation."""
    def __init__(self, name): self.name = name
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
