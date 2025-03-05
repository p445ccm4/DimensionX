import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image
import moviepy as mp
import argparse
import src.gradio_demo.utils as utils
from src.gradio_demo.rife_model import load_rife_model, rife_inference_with_latents
from diffusers.image_processor import VaeImageProcessor

class DimensonX:
    def __init__(self, cogvideox_path="./models/CogVideoX-5b-I2V", lora_path="./models/model_DimensionX", realesrgan_path="./models/model_real_esran/RealESRGAN_x4.pth", rife_path="./models/model_rife"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(cogvideox_path, torch_dtype=torch.bfloat16)
        lora_rank = 256
        self.pipe.load_lora_weights(lora_path, weight_name="orbit_left_lora_weights.safetensors", adapter_name="test_1")
        self.pipe.fuse_lora(lora_scale=1 / lora_rank)
        self.pipe.to(self.device)
        self.upscale_model = utils.load_sd_upscale(realesrgan_path, self.device)
        self.frame_interpolation_model = load_rife_model(rife_path)
        self.vae_processor = VaeImageProcessor() # Initialize VaeImageProcessor here

    def img2video(self, prompt, image, output_video_path, scale=False, interpolate=False):
        # Horizontally flip the image
        image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

        video = self.pipe(
            image,
            prompt,
            use_dynamic_cfg=True,
            output_type="pt",
            num_inference_steps=50,
            guidance_scale=7.0,
        ).frames
        video_flipped = self.pipe(
            image_flipped,
            prompt,
            use_dynamic_cfg=True,
            output_type="pt",
            num_inference_steps=50,
            guidance_scale=7.0,
        ).frames

        if scale:
            video = utils.upscale_batch_and_concatenate(self.upscale_model, video, self.device)
            video_flipped = utils.upscale_batch_and_concatenate(self.upscale_model, video_flipped, self.device)

        if interpolate:
            video = rife_inference_with_latents(self.frame_interpolation_model, video)
            video = rife_inference_with_latents(self.frame_interpolation_model, video)
            video_flipped = rife_inference_with_latents(self.frame_interpolation_model, video_flipped)
            video_flipped = rife_inference_with_latents(self.frame_interpolation_model, video_flipped)

        # Export the video
        video_path_right = output_video_path.replace(".mp4", "_right.mp4")
        video_path_left = output_video_path.replace(".mp4", "_left.mp4")

        video_numpy = self.vae_processor.pt_to_numpy(video[0])
        video_pil = self.vae_processor.numpy_to_pil(video_numpy)
        export_to_video(video_pil, video_path_right, fps=16)

        video_flipped_numpy = self.vae_processor.pt_to_numpy(video_flipped[0])
        video_flipped_pil = self.vae_processor.numpy_to_pil(video_flipped_numpy)
        export_to_video(video_flipped_pil, video_path_left, fps=16)


        # Load the video, flip it horizontally, and save
        clip_left = mp.VideoFileClip(video_path_left).with_effects([mp.vfx.MirrorX(), mp.vfx.TimeMirror()]).subclipped(0.53, -1)
        clip_right = mp.VideoFileClip(video_path_right).subclipped(0, 1.53)

        # Concatenate the flipped video with the other video
        final_clip = mp.concatenate_videoclips([clip_left, clip_right])

        # Export the final concatenated video
        final_clip.write_videofile(output_video_path, fps=16)
        # final_clip.write_gif(output_video_path.replace(".mp4", ".gif"), fps=16)

        # Clean up temporary files
        clip_left.close()
        clip_right.close()
        final_clip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from image using CogVideoX")
    parser.add_argument("--prompt", type=str, default="interior design", help="Prompt for video generation")
    parser.add_argument("--input_image", type=str, default="inputs/a1.jpg", help="Path to input image")
    parser.add_argument("--output_video", type=str, default="outputs/a1.mp4", help="Path for output video")
    parser.add_argument("--scale", action="store_true", help="Enable upscaling of the video")
    parser.add_argument("--interpolate", action="store_true", help="Enable RIFE interpolation on the video")
    args = parser.parse_args()

    generator = DimensonX()
    generator.img2video(args.prompt, args.input_image, args.output_video, args.scale, args.interpolate)

    print(f"Video saved to {args.output_video}")