from transformers import PretrainedConfig

class KrakenConfig(PretrainedConfig):
    model_type = "kraken"
    
    def __init__(self, config_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config_dict or {}
