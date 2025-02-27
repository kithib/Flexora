export CUDA_VISIBLE_DEVICES=0
python run.py --datasets your_dataset --hf-type base --hf-path /path to base model --peft-path /path to peft model --num-gpus 1 --debug
