from motionnet.models.ptr.ptr import PTR

__all__ = {
    'ptr': PTR,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
