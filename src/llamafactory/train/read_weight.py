from transformers import AutoModelForCausalLM

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    "/apdcephfs_cq10/share_1567347/kitwei/LLaMA-Factory/saves/llama3-8b/delete_test/delete_model", 
    device_map="auto", 
    trust_remote_code=True
)
new_state_dict = model.state_dict().pop('lm_head.weight')
# 打印模型结构，确认路径
print(new_state_dict)
