from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')

pipeline.load_lora_weights('artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl', weight_name='PS1Redmond-PS1Game-Playstation1Graphics.safetensors')

a = input('Prompt: ').strip()

image = pipeline(str(a)).images[0]

image.save('test.png')