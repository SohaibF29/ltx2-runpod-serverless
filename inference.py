import torch
from huggingface_hub import hf_hub_download, snapshot_download
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_core.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig, get_video_chunks_number

__all__ = [
    'calculate_video_config',
    'TI2VidTwoStagesPipeline',
    'LoraPathStrengthAndSDOps',
    'LTXV_LORA_COMFY_RENAMING_MAP',
    'encode_video',
    'AUDIO_SAMPLE_RATE',
    'TilingConfig',
    'SpatialTilingConfig',
    'TemporalTilingConfig',
    'get_video_chunks_number',
    'hf_hub_download',
    'snapshot_download'
]

def calculate_video_config(
    duration_seconds: float,
    orientation: str = "landscape",
    frame_rate: float = 24.0,
    base_resolution: int = 1080,
    num_inference_steps: int = 50,
    cfg_guidance_scale: float = 4.5,
    seed: int = 42,
) -> dict:
    """
    Calculate video configuration from simple parameters.

    Args:
        duration_seconds: Target duration in seconds (e.g., 8.0, 30.0)
        orientation: "portrait" (9:16), "landscape" (16:9), or "square" (1:1)
        frame_rate: Frames per second (default 24) - USER CONFIGURABLE
        base_resolution: Base size in pixels (default 1080). 
                        For landscape: this is height
                        For portrait: this is width
                        For square: this is both
        num_inference_steps: Diffusion steps
        cfg_guidance_scale: CFG scale
        seed: Random seed

    Returns:
        Dictionary with all video configuration parameters
    """

    # Calculate frames: must be 1 + 8*k for LTX VAE compatibility
    raw_frames = duration_seconds * frame_rate
    k = max(0, round((raw_frames - 1) / 8))
    num_frames = 1 + 8 * k

    actual_duration = num_frames / frame_rate
    print(f"Requested: {duration_seconds}s @ {frame_rate}fps, Actual: {actual_duration:.2f}s ({num_frames} frames)")

    # Calculate dimensions based on orientation
    # Must be divisible by 64 for two-stage pipeline
    divisor = 64

    if orientation.lower() == "landscape":
        # 16:9 aspect ratio (width:height)
        height = (base_resolution // divisor) * divisor
        width = int(height * 16 / 9)
        width = (width // divisor) * divisor

    elif orientation.lower() == "portrait":
        # 9:16 aspect ratio (width:height)
        width = (base_resolution // divisor) * divisor
        height = int(width * 16 / 9)  # height is longer in portrait
        height = (height // divisor) * divisor

    elif orientation.lower() == "square":
        # 1:1 aspect ratio
        size = (base_resolution // divisor) * divisor
        width = height = size

    else:
        raise ValueError(f"Orientation must be 'portrait', 'landscape', or 'square', got: {orientation}")

    print(f"Orientation: {orientation}, Resolution: {width}x{height} ({width/height:.3f})")

    return {
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "cfg_guidance_scale": cfg_guidance_scale,
        "seed": seed,
    }


def main():
    # ==================== BEST QUALITY CONFIGURATION ====================

    # Use the helper to compute the video configuration.
    # Matches the previous hard-coded settings (8s @ 24fps, 1088 reference height, landscape)
    video_config = calculate_video_config(
        duration_seconds=25,
        orientation="portrait",
        frame_rate=12,
        base_resolution=1088,
        num_inference_steps=50,
        cfg_guidance_scale=4.5,
        seed=42,
    )

    generation_config = {
        "prompt": (
            "A majestic golden retriever puppy playing joyfully in a sunlit meadow filled with vibrant wildflowers, butterflies fluttering around, and a gentle stream flowing nearby, while a young girl in a flowing white dress laughs and runs alongside, her hair catching the sunlight, captured in a smooth cinematic slow-motion, with dynamic camera angles slowly panning and tilting to follow the action, soft lens flare from the morning sun, warm and dreamy lighting, subtle depth of field focusing on the puppy and girl, cinematic color grading with rich warm tones, light motion blur for realism, a soft mist rising from the stream, gentle floating pollen and dust particles, an uplifting orchestral soundtrack with flutes and strings syncing with the movement, occasional birds chirping and water sounds, ultra-photorealistic, hyper-detailed textures, emotional emphasis and immersive visual richness."
        ),
        "negative_prompt": (
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
            "excessive noise, grainy texture, flickering, motion blur, distorted proportions, "
            "artifacts, cartoonish rendering, AI artifacts, deformed, mutated, extra limbs, "
            "wrong anatomy, bad hands, bad face, asymmetrical, unnatural poses, text, watermarks, "
            "signature, logo, jpeg artifacts, compression errors, oversaturated, undersaturated, "
            "tiling, duplicate objects, cropped, low resolution, pixelated, glitch, unnatural lighting, "
            "shadow errors, color banding, unrealistic reflections, cluttered composition"
        ),
        "enhance_prompt": False,
        "output_path": "new_video.mp4",
    }

    images = []

    # ==================== MODEL DOWNLOAD ====================

    print("Downloading LTX-2 models from Hugging Face Hub...")
    model_paths = {
        "checkpoint_path": hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-19b-dev.safetensors",
            cache_dir="./models"
        ),
        "upsampler_path": hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            cache_dir="./models"
        ),
        "distilled_lora_path": hf_hub_download(
            repo_id="Lightricks/LTX-2",
            filename="ltx-2-19b-distilled-lora-384.safetensors",
            cache_dir="./models"
        ),
        "gemma_root": snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            cache_dir="./models",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        ),
    }

    # ==================== DEVICE & MEMORY ====================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f} GB")
        use_fp8 = total_mem < 32
        if use_fp8:
            print("Enabling FP8 mode")
    else:
        use_fp8 = False

    # ==================== TILING CONFIGURATION ====================
    # CRITICAL FIX: Always enable tiling for high resolutions to avoid 
    # "input tensor must fit into 32-bit index math" error
    # At 1920x1088x192 frames, the VAE decoder produces tensors with >2^31 elements

    print("Configuring VAE tiling for high resolution...")

    # For 1080p resolution, we use smaller spatial tiles to ensure no single 
    # convolution operation exceeds the 32-bit indexing limit (2,147,483,647 elements)
    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=320,      # Reduced from default 512 for 1080p safety
            tile_overlap_in_pixels=64
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=64,       # Default is usually fine for temporal
            tile_overlap_in_frames=24
        )
    )

    # Calculate chunks for progress reporting
    video_chunks_number = get_video_chunks_number(
        video_config["num_frames"], tiling_config
    )

    print(f"Tiling enabled: {video_chunks_number} chunk(s) for {video_config['num_frames']} frames")

    # ==================== PIPELINE INIT ====================

    print("Initializing TI2VidTwoStagesPipeline...")

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=model_paths["distilled_lora_path"],
            strength=0.6,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    # ==================== CUSTOM LORAs ====================

    print("Downloading LoRAs from Hugging Face...")

    lora_paths = {
        "depth": hf_hub_download(
            repo_id="Lightricks/LTX-2-19b-IC-LoRA-Depth-Control",
            filename="ltx-2-19b-ic-lora-depth-control.safetensors",
            cache_dir="./models/loras",
        ),
        "detailer": hf_hub_download(
            repo_id="Lightricks/LTX-2-19b-IC-LoRA-Detailer",
            filename="ltx-2-19b-ic-lora-detailer.safetensors",
            cache_dir="./models/loras",
        ),
        "camera_static": hf_hub_download(
            repo_id="Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
            filename="ltx-2-19b-lora-camera-control-static.safetensors",
            cache_dir="./models/loras",
        ),
    }

    loras = [
        # Structural realism
        LoraPathStrengthAndSDOps(
            path=lora_paths["depth"],
            strength=0.6,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ),
        LoraPathStrengthAndSDOps(
            path=lora_paths["detailer"],
            strength=0.48,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ),
        LoraPathStrengthAndSDOps(
            path=lora_paths["camera_static"],
            strength=0.7,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=model_paths["checkpoint_path"],
        distilled_lora=distilled_lora,
        spatial_upsampler_path=model_paths["upsampler_path"],
        gemma_root=model_paths["gemma_root"],
        loras=loras,
        device=device,
        fp8transformer=use_fp8,
    )

    print(f"""
    Generating {video_config['num_frames']} frames @ {video_config['frame_rate']}fps
    Resolution: {video_config['width']}x{video_config['height']}
    Duration: ~{video_config['num_frames'] / video_config['frame_rate']:.1f}s
    Inference steps: {video_config['num_inference_steps']}
    CFG scale: {video_config['cfg_guidance_scale']}
    Chunks: {video_chunks_number}
    """)

    with torch.no_grad():
        video_iterator, audio = pipeline(
            prompt=generation_config["prompt"],
            negative_prompt=generation_config["negative_prompt"],
            seed=video_config["seed"],
            height=video_config["height"],
            width=video_config["width"],
            num_frames=video_config["num_frames"],
            frame_rate=video_config["frame_rate"],
            num_inference_steps=video_config["num_inference_steps"],
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=video_config["cfg_guidance_scale"],
                stg_scale=0.0,
                rescale_scale=0.0,
                modality_scale=1.0,
                skip_step=0,
                stg_blocks=[],
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=video_config["cfg_guidance_scale"],
                stg_scale=0.0,
                rescale_scale=0.0,
                modality_scale=1.0,
                skip_step=0,
                stg_blocks=[],
            ),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=generation_config["enhance_prompt"],
        )

        print(f"Encoding video to {generation_config['output_path']}...")
        encode_video(
            video=video_iterator,
            fps=video_config["frame_rate"],
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=generation_config["output_path"],
            video_chunks_number=video_chunks_number,
        )

    print(f"\n✓ Video saved to {generation_config['output_path']}")

    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()