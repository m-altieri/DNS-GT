class DummyStrategy:
    class _ContextManager:
        def __enter__(self):
            print(
                "[Experimental] Using a dummy distribution strategy (no distribution)"
            )

        def __exit__(self, *args):
            pass

    @classmethod
    def scope(cls):
        return cls._ContextManager()
