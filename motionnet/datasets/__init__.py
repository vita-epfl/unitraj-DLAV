from .ptr_dataset import PTRDataset

__all__ = {
    'ptr': PTRDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
