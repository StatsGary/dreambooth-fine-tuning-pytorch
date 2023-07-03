""" 
Name:       dreambooth_train.py
Author:     Gary Hutson
Date:       23/05/2023
Usage:      python dreambooth_train.py
"""

from dreambooth.dataloader import pull_dataset_from_hf_hub, DreamBoothDataset
from dreambooth.image import image_grid
from dreambooth.collator import collate_fn
from dreambooth.train import train_dreambooth
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel
import logging
import yaml

# SET project constants and variables
with open('dreambooth_param.yml', 'r') as train:
    params = yaml.safe_load(train)

# Set the parameters
train_params = params['train_params']
STABLE_DIFFUSION_NAME = train_params['stable_diffusion_backbone']
FEATURE_EXTRACTOR = train_params['feature_extractor']
hf_data_location = train_params['hugging_face_image_store']
learning_rate = float(train_params['learning_rate'])
max_train_steps = int(train_params['max_train_steps'])
resolution = int(train_params['resolution'])
train_batch_size=int(train_params['train_bs'])
grad_accum_steps=int(train_params['grad_accum_steps'])
max_gradient_norm=float(train_params['max_gradient_norm'])
sample_batch_size=int(train_params['sample_batch_size'])
model_checkpoint_name=str(train_params['model_checkpoint_name'])
shuffle_train=bool(train_params['random_shuffle_train_set'])
use_8bit_optim=bool(train_params['use_8bit_optimizer'])
name_of_your_concept=train_params['concept_name']
object_type=train_params['item_type']


if __name__ =='__main__':
    # Load the image dataset from HuggingFace hub
    dataset = pull_dataset_from_hf_hub(dataset_id=hf_data_location)

    # Name your concept and set of images
    name_of_your_concept = name_of_your_concept
    type_of_thing = object_type
    instance_prompt = f"a photo of {name_of_your_concept} {type_of_thing}"
    print(f"Instance prompt: {instance_prompt}")

    # Load the CLIP tokenizer
    model_id = STABLE_DIFFUSION_NAME
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer")

    # Create a train dataset from the Dreambooth data loader
    train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)

    # Get text encoder, UNET and VAE
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR)

    # Train the model
    model = train_dreambooth(
        text_encoder=text_encoder, 
        vae = vae, 
        unet = unet, 
        tokenizer=tokenizer, 
        feature_extractor=feature_extractor, 
        train_dataset=train_dataset, 
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps, 
        shuffle_train=shuffle_train,
        gradient_accumulation_steps=grad_accum_steps, 
        use_8bit_ADAM=True, 
        learning_rate=learning_rate, 
        max_grad_norm=max_gradient_norm,
        output_dir=model_checkpoint_name
    )