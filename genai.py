import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import login
import os
import argparse

# Log in to Hugging Face (replace the token with your own)
login(token="Your Token")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images for Chest X-ray labels")
    parser.add_argument("--model_path", type=str, default="roentgen", 
                        help="Choose one of: 'roentgen', 'sd1.4', 'sd1.5', 'sd2.0', 'sdxl' (default: 'roentgen')")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image (default: 512)")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt call (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference: 'cuda' or 'cuda:0', 'cuda:1', etc. (default: 'cuda')")
    parser.add_argument("--iteration", type=int, default=1000, help="Number of iterations/images to generate (default: 1)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine the actual model path and pipeline class
    if args.model_path == 'roentgen':
        model_path = './roentgen'
        pipeline_class = StableDiffusionPipeline
        scheduler = None
    elif args.model_path == 'sd1.4':
        model_path = 'CompVis/stable-diffusion-v1-4'
        pipeline_class = StableDiffusionPipeline
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif args.model_path == 'sd1.5':
        model_path = 'runwayml/stable-diffusion-v1-5'
        pipeline_class = StableDiffusionPipeline
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif args.model_path == 'sd2.0':
        model_path = 'stabilityai/stable-diffusion-2-1-base'
        pipeline_class = StableDiffusionPipeline
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif args.model_path == 'sdxl':
        model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
        pipeline_class = StableDiffusionXLPipeline
        scheduler = None
    else:
        print("Invalid model_path option provided. Exiting.")
        exit(1)

    label_mapping = {
        0: "atelectasis",
        1: "cardiomegaly",
        2: "effusion",
        3: "infiltration",
        4: "mass",
        5: "nodule",
        6: "pneumonia",
        7: "pneumothorax",
        8: "consolidation",
        9: "edema",
        10: "emphysema",
        11: "fibrosis",
        12: "pleural",
        13: "hernia",
    }

    output_dir = f"./synthetic_data/{args.model_path}"
    os.makedirs(output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU installation.")
        exit(1)

    print(f"Using device: {args.device}")

    pipe = pipeline_class.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(args.device)

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    for label_id, label_text in label_mapping.items():
        print(f"Generating images for label {label_id} ('{label_text}')...")
        label_dir = os.path.join(output_dir, label_text)
        os.makedirs(label_dir, exist_ok=True)

        prompt = f"Frontal Chest X-ray with {label_text}"

        for iter_idx in range(args.iteration):
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = pipe(
                    prompt,
                    num_inference_steps=75,
                    height=args.height,
                    width=args.width,
                    guidance_scale=4,
                    num_images_per_prompt=args.num_images_per_prompt
                )

            for i, image in enumerate(output.images):
                image_path = os.path.join(label_dir, f"{iter_idx}.png")
                image.save(image_path)
                print(f"  Saved image {iter_idx+1}/{args.iteration} for '{label_text}'")

    print("All images generated and saved.")

if __name__ == "__main__":
    main()