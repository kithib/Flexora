# Flexora
Our code is developed based on Llama-factory:
https://github.com/hiyouga/LLaMA-Factory.git

Our evaluation framework is opencompass:
https://github.com/open-compass/opencompass.git

## Flexora introduction
Large Language Models (LLMs) are driving advancements in artificial intelligence by increasing the scale of model parameters, which has significantly enhanced generalization ability and unlocked new capabilities in practice. However, their performance in specific downstream tasks is usually hindered by their knowledge boundaries on these tasks. Thus, fine-tuning techniques, especially the widely used Low-Rank Adaptation(LoRA) method, have been introduced to expand the boundaries on these tasks, whereas LoRA would underperform on certain tasks owing to its potential overfitting on these tasks. To overcome this overfitting and improve the performance of LoRA, we propose the flexible low rank adaptation (Flexora) method to automatically and flexibly select the most important layers needing to be fine-tuned to achieve the best performance on different downstream tasks. Specifically, Flexora firstly frames this layer selection problem as a well-defined hyperparameter optimization (HPO) problem, then addresses
it using the unrolled differentiation (UD) method, and finally selects the most useful layers based on the optimized hyperparameters. Our extensive experiments on many pre-trained models and natural language tasks show that Flexora is able to consistently improve over the existing baselines, indicatingmthe effectiveness of our Flexora in practice. We additionally provide insightful theoretical results and many ablation studies to deliver a comprehensive understanding of our Flexora.

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
cd flexora/LLaMA-Factory
bash flexora.sh # Flexible Layer Selection Stage

#Modify flexora/LLaMA-Factory/src/llamafactory/train/sft/workflow.py to change the selected layer for fine-tuning
bash run.sh # Fine-Tuning Stage

cd ../opencompass
bash eval.sh #Evaluation
```