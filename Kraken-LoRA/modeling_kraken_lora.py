import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TextClassificationPipeline
from configuration_kraken_lora import KrakenConfig
import tokenizer_template_switch
from peft import PeftModel, PeftConfig  # Import necessary modules for LoRA

class KrakenForCausalLM(PreTrainedModel):
    config_class = KrakenConfig

    def __init__(self, config):
        super().__init__(config)
        self.tokenizers = {key: AutoTokenizer.from_pretrained(name, device_map="auto") for key, name in config.config_dict['tokenizers'].items()}
        self.model = self.load_base_model(config.config_dict['models']['base'], config.config_dict['quantization']['base'])  # Load only expert1 as the base model
        self.lora_adapters = config.config_dict['lora_adapters']  # Load LoRA adapter paths
        self.router_model = AutoModelForSequenceClassification.from_pretrained(config.config_dict['router'], trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(config.config_dict['router'], trust_remote_code=True, device_map="auto")
        self.router = TextClassificationPipeline(model=self.router_model, tokenizer=self.tokenizer)
        self.models_indices = config.config_dict['class_indices']

    def load_base_model(self, model_name, quantization):
        if quantization == "8bit":
            return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", load_in_8bit=True, torch_dtype="auto")
        elif quantization == "4bit":
            return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", load_in_4bit=True, torch_dtype="auto")
        elif quantization == "awq":
            return self.load_awq_model(model_name)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto")

    def load_awq_model(self, name):
        return AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto")

    def load_lora_adapter(self, base_model, adapter_path):
        print("Loading adapter: "+adapter_path)
        return PeftModel.from_pretrained(base_model, adapter_path)

    def tokenize_inputs(self, text, adapter_key):
        return self.tokenizers[adapter_key](text, return_tensors="pt")

    def determine_adapter(self, text):
        prediction = self.router(text)[0]["label"]
        model_decision_index = self.models_indices[prediction]
        adapter_keys = ['lora_expert1', 'lora_expert2', 'lora_expert3', 'lora_expert4', 'lora_expert5']
        return adapter_keys[model_decision_index]

    def expert_tokenizer(self, text):
        adapter_key = self.determine_adapter(text)
        return self.tokenizers[adapter_key]


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


        # Determine the appropriate LoRA adapter
        adapter_key = self.determine_adapter(text)
        print(f"Choosing LoRA adapter for {adapter_key} ..")
        # Load and apply the LoRA adapter to the base model (expert1)
        lora_adapter_path = self.lora_adapters[adapter_key]
        model_with_lora = self.load_lora_adapter(self.model, lora_adapter_path)

        # Use the tokenizer for the selected expert to tokenize the inputs
        mod_txt = self.tokenizers[adapter_key].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        current_device = input_ids.device if isinstance(input_ids, torch.Tensor) else 'cpu'
        
        # Tokenize accordingly to the best model

        tok = self.tokenizers[adapter_key](mod_txt, return_tensors="pt")
        tok_input_ids = tok.input_ids.to(current_device)
        tok_attention_mask = tok.attention_mask.to(current_device)

        # Generate text using the modified model
        return model_with_lora.generate(tok_input_ids, attention_mask=tok_attention_mask, **generate_kwargs)
    

      