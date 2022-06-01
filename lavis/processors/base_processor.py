class BaseProcessor:
    def __init__(self):
        return

    def __call__(self, item):
        return item

    @classmethod
    def build_from_cfg(cls, cfg=None):
        return cls()
