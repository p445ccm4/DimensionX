import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image
import moviepy as mp
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description="Generate video from image using CogVideoX")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for video generation")
parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
parser.add_argument("--output_video", type=str, required=True, help="Path for output video")
args = parser.parse_args()

prompt = args.prompt
input_image_path = args.input_image
output_video_path = args.output_video
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
lora_path = "./models/model_DimensionX"
lora_rank = 256
pipe.load_lora_weights(lora_path, weight_name="orbit_left_lora_weights.safetensors", adapter_name="test_1")
pipe.fuse_lora(lora_scale=1 / lora_rank)
pipe.to("cuda")

image = load_image(input_image_path)

# Rest of your code remains the same...

# Horizontally flip the image
image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

video = pipe(image, prompt, use_dynamic_cfg=True)
video_flipped = pipe(image_flipped, prompt, use_dynamic_cfg=True)

# Export the video
video_path_right = output_video_path.replace(".mp4", "_right.mp4")
video_path_left = output_video_path.replace(".mp4", "_left.mp4")
export_to_video(video.frames[0], video_path_right, fps=8)
export_to_video(video_flipped.frames[0], video_path_left, fps=8)

# Load the video, flip it horizontally, and save
clip_left = mp.VideoFileClip(video_path_left).with_effects([mp.vfx.MirrorX(), mp.vfx.TimeMirror()]).subclipped(0, -1)
clip_right = mp.VideoFileClip(video_path_right)

# Concatenate the flipped video with the other video
final_clip = mp.concatenate_videoclips([clip_left, clip_right])

# Export the final concatenated video
final_clip.write_videofile(output_video_path)

# Clean up temporary files
clip_left.close()
clip_right.close()
final_clip.close()