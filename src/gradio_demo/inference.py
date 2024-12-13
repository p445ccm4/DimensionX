import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image
import moviepy as mp
import argparse
import utils
from rife_model import load_rife_model, rife_inference_with_latents
from diffusers.image_processor import VaeImageProcessor

# Add argument parsing
parser = argparse.ArgumentParser(description="Generate video from image using CogVideoX")
parser.add_argument("--prompt", type=str, default="interior design", help="Prompt for video generation")
parser.add_argument("--input_image", type=str, default="inputs/a1.jpg", help="Path to input image")
parser.add_argument("--output_video", type=str, default="outputs/a1.mp4", help="Path for output video")
parser.add_argument("--scale", action="store_true", help="Enable upscaling of the video")
parser.add_argument("--interpolate", action="store_true", help="Enable RIFE interpolation on the video")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = args.prompt
input_image_path = args.input_image
output_video_path = args.output_video
pipe = CogVideoXImageToVideoPipeline.from_pretrained("./models/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
lora_path = "./models/model_DimensionX"
lora_rank = 256
pipe.load_lora_weights(lora_path, weight_name="orbit_left_lora_weights.safetensors", adapter_name="test_1")
pipe.fuse_lora(lora_scale=1 / lora_rank)
pipe.to(device)

upscale_model = utils.load_sd_upscale("./models/model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("./models/model_rife")

image = load_image(input_image_path)

# Horizontally flip the image
image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

video = pipe(
    image,
    prompt,
    use_dynamic_cfg=True,
    output_type="pt",
    num_inference_steps=50,
    guidance_scale=7.0,
).frames
video_flipped = pipe(
    image_flipped, 
    prompt, 
    use_dynamic_cfg=True,
    output_type="pt",
    num_inference_steps=50,
    guidance_scale=7.0,
).frames
if args.scale:
# if True:
    video = utils.upscale_batch_and_concatenate(upscale_model, video, device)
    video_flipped = utils.upscale_batch_and_concatenate(upscale_model, video_flipped, device)
if args.interpolate:
# if True:
    video = rife_inference_with_latents(frame_interpolation_model, video)
    video_flipped = rife_inference_with_latents(frame_interpolation_model, video_flipped)

# Export the video
video_path_right = output_video_path.replace(".mp4", "_right.mp4")
video_path_left = output_video_path.replace(".mp4", "_left.mp4")
video = VaeImageProcessor.pt_to_numpy(video[0])
video = VaeImageProcessor.numpy_to_pil(video)
video_flipped = VaeImageProcessor.pt_to_numpy(video_flipped[0])
video_flipped = VaeImageProcessor.numpy_to_pil(video_flipped)
export_to_video(video, video_path_right, fps=16)
export_to_video(video_flipped, video_path_left, fps=16)

# Load the video, flip it horizontally, and save
clip_left = mp.VideoFileClip(video_path_left).with_effects([mp.vfx.MirrorX(), mp.vfx.TimeMirror()]).subclipped(0, -1)
clip_right = mp.VideoFileClip(video_path_right)

# Concatenate the flipped video with the other video
final_clip = mp.concatenate_videoclips([clip_left, clip_right])

# Export the final concatenated video
final_clip.write_videofile(output_video_path)
final_clip.write_gif(output_video_path.replace(".mp4", ".gif"), fps=16)

# Clean up temporary files
clip_left.close()
clip_right.close()
final_clip.close()