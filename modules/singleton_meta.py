class SingletonMeta(type):
    """
    싱글톤 메타클래스를 사용하여 클래스를 싱글톤으로 만듭니다.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
