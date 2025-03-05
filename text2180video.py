import deepseek, dimensionX, flux

def Text2180video(init_msg):
    _deepseek = deepseek.DeepSeek()
    _dimensionX = dimensionX.DimensonX()
    _flux = flux.Flux()

    prompt = _deepseek.gen_prompt(init_msg)
    print(prompt)
    image = _flux.gen_image(prompt)
    _dimensionX.img2video(
        prompt=prompt, 
        image=image, 
        output_video_path="outputs/text2180video_c0.mp4", 
        scale=False, 
        interpolate=True
    )

init_msg = "複式單位，可以坐在黑色沙發上看電視。"

Text2180video(init_msg)
print("done")