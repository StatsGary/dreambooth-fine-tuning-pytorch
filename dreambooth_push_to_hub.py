""" 
Name:       dreambooth_push_to_hub.py
Author:     Gary Hutson
Date:       23/05/2023
Usage:      python dreambooth_push_to_hub.py
"""

import yaml
import argparse as AP
from huggingface_hub import HfApi, ModelCard, create_repo, get_full_repo_name

# Load settings from YAML file
with open('dreambooth_param.yml', 'r') as train:
    params = yaml.safe_load(train)

train_params = params['train_params']

if __name__ =='__main__':
    model_card_description ="""
    
    """
    name_of_your_concept = train_params['concept_name'] 
    type_of_thing = train_params['item_type']
    model_name = f"{name_of_your_concept}-{type_of_thing}"
    description = f"""
    This is a Stable Diffusion model fine-tuned on `{type_of_thing}` images for the {type_of_thing} theme.
    """
    hub_model_id = get_full_repo_name(model_name)
    create_repo(hub_model_id)
    api = HfApi()
    api.upload_folder(folder_path=train_params['model_checkpoint_name'], path_in_repo="", repo_id=hub_model_id)
    content = model_card_description
    card = ModelCard(content)
    hub_url = card.push_to_hub(hub_model_id)
    print(f"Upload successful! Model can be found here: {hub_url}")