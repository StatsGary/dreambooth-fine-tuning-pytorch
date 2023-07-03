from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

def push_data_to_hf_hub(dataset_name,local_data_dir):
    dataset = load_dataset("imagefolder", data_dir=local_data_dir)
    # Remove the dummy label column
    dataset = dataset.remove_columns("label")
    # Push to Hub
    dataset.push_to_hub(dataset_name)

def pull_dataset_from_hf_hub(dataset_id='StatsGary/dreambooth-hackathon-images'):
    dataset_id = dataset_id
    dataset = load_dataset(dataset_id, split="train")
    print(f"Loaded dataset number of rows: {len(dataset)}")
    return dataset


class DreamBoothDataset(Dataset):
    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = self.dataset[index]["image"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example