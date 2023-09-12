import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


class VideoDiffusor:
    def __init__(
        self,
        num_inference_steps: int = 40,
        height=320,
        width=576,
        num_frames: int = 24
    ):
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames


        self.pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w", torch_dtype=torch.float16
        )
        self.pipe = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def create(self, prompt):
        video_frames = self.pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames
        ).frames
        
        video_path = export_to_video(video_frames)
        return video_path
    

class ImageDiffusor:
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to("cuda")
    
    def create(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        return images
