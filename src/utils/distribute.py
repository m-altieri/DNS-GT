class DummyStrategy:
    class _ContextManager:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    @classmethod
    def scope(cls):
        return cls._ContextManager()
