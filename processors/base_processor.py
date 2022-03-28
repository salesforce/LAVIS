class BaseProcessor:
    def __init__(self):
        return

    def __call__(self, item):
        raise NotImplementedError

    @classmethod
    def build_processor(cls, cfg):
        return cls()
