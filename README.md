# Flexora

Our code is developed based on **Llama-factory**:  
[https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)  

Our evaluation framework is **OpenCompass**:  
[https://github.com/open-compass/opencompass.git](https://github.com/open-compass/opencompass.git)  

## Introduction to Flexora

Large Language Models (LLMs) are driving advancements in artificial intelligence by increasing the scale of model parameters, which has significantly enhanced generalization ability and unlocked new capabilities in practice. However, their performance in specific downstream tasks is often hindered by their knowledge boundaries on these tasks. To address this, fine-tuning techniques, especially the widely used **Low-Rank Adaptation (LoRA)** method, have been introduced to expand these boundaries. However, LoRA can underperform on certain tasks due to potential overfitting.  

To overcome this limitation and improve the performance of LoRA, we propose **Flexible Low-Rank Adaptation (Flexora)**, a method that automatically and flexibly selects the most important layers to fine-tune for optimal performance on different downstream tasks. Specifically, Flexora:  
1. Frames the layer selection problem as a well-defined **Hyperparameter Optimization (HPO)** problem.  
2. Addresses it using the **Unrolled Differentiation (UD)** method.  
3. Selects the most useful layers based on the optimized hyperparameters.  

Extensive experiments on various pre-trained models and natural language tasks demonstrate that Flexora consistently outperforms existing baselines, highlighting its effectiveness in practice. Additionally, we provide insightful theoretical results and ablation studies to offer a comprehensive understanding of Flexora.  

## File Structure

- **src**: Contains the implementation of Flexora's optimizer and related modifications.  
  - `src/llamafactory/train/flora.py`: Flexora Optimizer.  
  - `src/llamafactory/train/trainer_utils.py`: Training code modifications.  
  - `src/llamafactory/train/sft/workflow.py`: Layer selection logic.  

## Getting Started

### Installation

```bash
mkdir flexora 
cd flexora
conda create --name flexora python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate flexora

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
cd ../..
pip install -r requirements.txt
```
### Then, follow these steps to update your code:
```bash
# Replace the necessary files and directories
rm- rf flexora/LLaMA-Factory/src
mv src flexora/LLaMA-Factory/src
mv flexora.yaml flexora/LLaMA-Factory/examples/train_lora/flexora.yaml
mv sft.yaml flexora/LLaMA-Factory/examples/train_lora/sft.yaml
mv flexora.sh flexora/LLaMA-Factory/flexora.sh
mv run.sh flexora/LLaMA-Factory/run.sh
mv eval.sh flexora/opencompass/eval.sh
```
### Now we can start training
```bash
# Navigate to the LLaMA-Factory directory and start training
cd flexora/LLaMA-Factory
bash flexora.sh  # Flexible Layer Selection Stage

# Modify the selected layers for fine-tuning
# Edit flexora/LLaMA-Factory/src/llamafactory/train/sft/workflow.py

bash run.sh  # Fine-Tuning Stage

# Navigate to OpenCompass and evaluate the model
cd ../opencompass
bash eval.sh  # Evaluation Stage
```
## Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{
wei2024flexora,
title={Flexora: Flexible Low-Rank Adaptation for Large Language Models},
author={Chenxing Wei and Yao Shu and Ying Tiffany He and Fei Richard Yu},
booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
year={2024},
url={https://openreview.net/forum?id=KIk3gM3xYd}
}
```

## Acknowledgement

This repository benefits from the following open-source projects:
- [Llama-factory](https://github.com/hiyouga/LLaMA-Factory.git) 
- [OpenCompass](https://github.com/open-compass/opencompass.git)

We sincerely thank the authors for their wonderful contributions.