""" 
Name:       dreambooth_eval.py
Author:     Gary Hutson
Date:       23/05/2023
Usage:      python dreambooth_eval.py prompt "a photo of a viking on a fjord"
"""

from dreambooth.image import image_grid
from diffusers import  StableDiffusionPipeline
import torch
import yaml
import argparse as AP

# Load settings from YAML file
with open('dreambooth_param.yml', 'r') as train:
    params = yaml.safe_load(train)

# Set the parameters
eval_params = params['eval_params']
train_params = params['train_params']
get_user_input = False

if __name__ == '__main__':
    # Load in fine tuned model
    model_name = train_params['model_checkpoint_name']
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Use an argument parser to test the model
    if get_user_input==False:
        prompt = str(eval_params['eval_prompt'])
    else:
        prompt = input()
    
    guidance_scale = 7

    num_cols = 2
    all_images = []
    for _ in range(num_cols):
        images = pipe(prompt, guidance_scale=guidance_scale).images
        all_images.extend(images)
    plt = image_grid(all_images, 1, num_cols)
    save_path = f"{eval_params['image_save_path']}/{str(prompt.replace(' ','')[-10:])}.jpg"
    plt.save(save_path)