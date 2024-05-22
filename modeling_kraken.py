import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TextClassificationPipeline
from configuration_kraken import KrakenConfig
import tokenizer_template_switch

class KrakenForCausalLM(PreTrainedModel):
    config_class = KrakenConfig

    def __init__(self, config):
        super().__init__(config)
        self.tokenizers = {key: AutoTokenizer.from_pretrained(name, device_map="auto") for key, name in config.config_dict['tokenizers'].items()}
        self.models = self.load_expert_models(config.config_dict['models'], config.config_dict['quantization'])
        self.router_model = AutoModelForSequenceClassification.from_pretrained(config.config_dict['router'], trust_remote_code=True,device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(config.config_dict['router'], trust_remote_code=True,device_map="auto")
        self.router = TextClassificationPipeline(model=self.router_model, tokenizer=self.tokenizer)
        self.models_indices = config.config_dict['class_indices']

    def load_expert_models(self, models_dict, quantization_dict):
        models = {}
        for key, name in models_dict.items():
            quantization = quantization_dict.get(key)
            if quantization == "8bit":
                models[key] = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto", load_in_8bit=True, torch_dtype="auto")
            elif quantization == "4bit":
                models[key] = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto", load_in_4bit=True, torch_dtype="auto")
            elif quantization == "awq":
                models[key] = self.load_awq_model(name)
            else:
                models[key] = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto", torch_dtype="auto")
        return models

    def load_awq_model(self, name):
        return AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")

    def tokenize_inputs(self, text, model_key):
        return self.tokenizers[model_key](text, return_tensors="pt")

    def determine_model(self, text):
        prediction = self.router(text)[0]["label"]
        model_decision_index = self.models_indices[prediction]
        model_keys = ['expert1', 'expert2', 'expert3', 'expert4','expert5']
        return model_keys[model_decision_index]
    
    def expert_tokenizer(self, text):
        model_key = self.determine_model(text)
        return self.tokenizers[model_key]


    def generate(self, input_ids, **generate_kwargs):
        # Tokenize the input_ids
        text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]

        msgs = tokenizer_template_switch.recover_chat_messages(text, self.tokenizer)
        if msgs and msgs[0]['role'] == 'system' and msgs[0]['content']=='<|im_start|>system':
            # Delete the first element
            msgs.pop(0)  
        # Check if the last element has the role 'assistant'
        if msgs and msgs[-1]['role'] == 'assistant':
            # Delete the last element
            msgs.pop()  

        # Determine the model key using the existing routing logic
        model_key = self.determine_model(text)
        # Show the routing result
        print(f"Choosing {model_key} ..")
        # Retrieve the model from the dictionary
        model = self.models[model_key]

        mod_txt = self.tokenizers[model_key].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        current_device = input_ids.device if isinstance(input_ids, torch.Tensor) else 'cpu'
        
        # Tokenize accordingly to the best model

        tok = self.tokenizers[model_key](mod_txt, return_tensors="pt")
        tok_input_ids = tok.input_ids.to(current_device)  
        tok_attention_mask = tok.attention_mask.to(current_device)

        # Generate text using the retrieved model
        return model.generate(tok_input_ids, attention_mask=tok_attention_mask, **generate_kwargs)
    

      