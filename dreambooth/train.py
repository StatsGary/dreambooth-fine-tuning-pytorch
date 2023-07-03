#Define training loop
import math
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
import torch
from dreambooth.collator import collate_fn


def train_dreambooth(text_encoder, vae, unet, tokenizer, feature_extractor, train_dataset, train_batch_size=1, max_train_steps=400, shuffle_train=True,
                        beta_start=0.00085, beta_end=0.012, beta_scheduler="scaled_linear", num_train_timesteps=1000, seed=3434554,
                        gradient_checkpoint=True, gradient_accumulation_steps=8, use_8bit_ADAM=True, 
                        learning_rate=2e-06, max_grad_norm=1.0, output_dir='stable-diffusion-trained'):

    # Takes the input from the training arguments to specify the warmup phase of the gradients
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    # Sets a reproduable seed to work 
    set_seed(seed)
    if gradient_checkpoint:
        unet.enable_gradient_checkpointing()

    if use_8bit_ADAM:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Then we implemenet and optimizer class which is used with the learning rate
    optimizer = optimizer_class(
        unet.parameters(),  # only optimize unet
        lr=learning_rate,
    )
    # Create a random noise scheduler to be applied to the images
    noise_scheduler = DDPMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_scheduler,
        num_train_timesteps=num_train_timesteps
    )

    # Pass the images into the training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description(f"Steps based on batch size {total_batch_size}")
    global_step = 0

    # Set the training loop for each epoch
    for epoch in range(num_train_epochs):
        print(f'Epoch: {epoch + 1} of {num_train_epochs}')
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        print(f"Loading pipeline and saving to {output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_scheduler,
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(output_dir)