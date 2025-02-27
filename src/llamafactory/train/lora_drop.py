import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
#export CUDA_VISIBLE_DEVICES=0
#nohup python /apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/src/llamafactory/train/lora_drop.py > /apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/log/lora_drop/layer_select_piqa.log &
class LoraDrop:
    def __init__(self, model_path, tokenizer_path, data_path, top_n):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data_path = data_path
        self.top_n = top_n
        self.model = None
        self.tokenizer = None
        self.data = None
        self.load_resources()

    def load_resources(self):
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Load the data
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

    def prepare_input_data(self, instruction_output_pair):
        # Concatenate instruction and output and prepare input data
        input_data = self.tokenizer(
            instruction_output_pair, return_tensors="pt", padding=True, truncation=True
        ).input_ids
        return input_data

    def analyze_hidden_layers(self):
        # Get model configuration
        config = self.model.config
        g_list = [0.0] * (config.num_hidden_layers + 1)

        # Ensure the model outputs hidden states
        self.model.config.output_hidden_states = True

        # Set the model to evaluation mode
        self.model.eval()

        # Perform inference and calculate the sum of squared L2 norms
        with torch.no_grad():
            for di in tqdm(self.data, desc="Analyzing", unit="entry"):
                input_data = self.prepare_input_data(di['instruction'] + di['output'])
                outputs = self.model(input_data)
                hidden_states = outputs.hidden_states
                for i, layer_output in enumerate(hidden_states):
                    g_list[i] += (torch.norm(layer_output, p=2).item() ** 2) / 1e10

        # Normalize and find top n indices
        g_list = g_list[1:]
        total_sum = sum(g_list)
        normalized_list = [element / total_sum for element in g_list]
        indexed_list = list(enumerate(normalized_list))
        indexed_list.sort(key=lambda x: x[1], reverse=True)
        top_n_indices = sorted([index for index, value in indexed_list[:self.top_n]])

        return top_n_indices

# Usage
# Assuming you have the correct paths for the model, tokenizer, and data
#model_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/flora_v1/hellaswag_qvproj_lora_sft"
#model_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/flora_v1/piqa_qvproj_lora_sft"
#model_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/flora_v1/race_qvproj_lora_sft"
model_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/flora_v1/winogrande_qvproj_lora_sft"
tokenizer_path = model_path  # Assuming the tokenizer is in the same directory as the model
#data_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/data/hellaswag_train.json"
#data_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/data/piqa_train.json"
#data_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/data/race_train.json"
data_path = "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/data/winogrande_train.json"
top_n = 16
lora_drop = LoraDrop(model_path, tokenizer_path, data_path, top_n)
top_n_indices = lora_drop.analyze_hidden_layers()
print(top_n_indices)