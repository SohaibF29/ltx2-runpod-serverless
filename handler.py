import os
import json
import runpod
import base64
import uuid
import asyncio
import tempfile
import traceback
import shutil
from io import BytesIO
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import boto3
import torch
from PIL import Image
import numpy as np

# Import functions from your inference.py
from inference import (
    calculate_video_config,
    TI2VidTwoStagesPipeline,
    LoraPathStrengthAndSDOps,
    LTXV_LORA_COMFY_RENAMING_MAP,
    encode_video,
    AUDIO_SAMPLE_RATE,
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
    get_video_chunks_number,
    hf_hub_download,
    snapshot_download
)

# Import the required guider params and defaults
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.utils.constants import (
    DEFAULT_VIDEO_GUIDER_PARAMS,
    DEFAULT_AUDIO_GUIDER_PARAMS
)

# AWS S3 Configuration
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "your-bucket-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = "videos/"

# Initialize S3 client (kept at import time; safe for most serverless deployments)
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

@dataclass
class GenerationRequest:
    """Data class for generation request parameters"""
    prompt: str
    negative_prompt: Optional[str] = None
    duration_seconds: float = 25.0
    orientation: str = "portrait"
    frame_rate: float = 12.0
    base_resolution: int = 1088
    num_inference_steps: int = 50
    
    # Video guidance parameters
    video_cfg_scale: float = 3.0
    video_stg_scale: float = 1.0
    video_rescale_scale: float = 0.7
    video_modality_scale: float = 3.0  # A2V guidance
    video_skip_step: int = 0
    video_stg_blocks: Optional[List[int]] = None
    
    # Audio guidance parameters - OPTIMIZED FOR QUALITY
    audio_cfg_scale: float = 7.0  # Higher CFG for stronger audio prompt adherence
    audio_stg_scale: float = 1.0  # Enable STG for temporal coherence
    audio_rescale_scale: float = 0.7
    audio_modality_scale: float = 3.0  # V2A guidance for lipsync
    audio_skip_step: int = 0
    audio_stg_blocks: Optional[List[int]] = None
    
    seed: int = 42
    enhance_prompt: bool = False
    image_data: Optional[str] = None
    image_format: str = "png"
    distilled_lora_strength: float = 0.6
    depth_lora_strength: float = 0.6
    detailer_lora_strength: float = 0.48
    camera_static_lora_strength: float = 0.7
    
    def __post_init__(self):
        # Set default STG blocks if not provided
        if self.video_stg_blocks is None:
            self.video_stg_blocks = [29]
        if self.audio_stg_blocks is None:
            self.audio_stg_blocks = [29]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationRequest':
        """Create request from dictionary"""
        return cls(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt"),
            duration_seconds=float(data.get("duration_seconds", 25.0)),
            orientation=data.get("orientation", "portrait"),
            frame_rate=float(data.get("frame_rate", 12.0)),
            base_resolution=int(data.get("base_resolution", 1088)),
            num_inference_steps=int(data.get("num_inference_steps", 50)),
            
            # Video params
            video_cfg_scale=float(data.get("video_cfg_scale", 3.0)),
            video_stg_scale=float(data.get("video_stg_scale", 1.0)),
            video_rescale_scale=float(data.get("video_rescale_scale", 0.7)),
            video_modality_scale=float(data.get("video_modality_scale", 3.0)),
            video_skip_step=int(data.get("video_skip_step", 0)),
            video_stg_blocks=data.get("video_stg_blocks", [29]),
            
            # Audio params - using higher defaults for quality
            audio_cfg_scale=float(data.get("audio_cfg_scale", 7.0)),
            audio_stg_scale=float(data.get("audio_stg_scale", 1.0)),
            audio_rescale_scale=float(data.get("audio_rescale_scale", 0.7)),
            audio_modality_scale=float(data.get("audio_modality_scale", 3.0)),
            audio_skip_step=int(data.get("audio_skip_step", 0)),
            audio_stg_blocks=data.get("audio_stg_blocks", [29]),
            
            seed=int(data.get("seed", 42)),
            enhance_prompt=bool(data.get("enhance_prompt", False)),
            image_data=data.get("image_data"),
            image_format=data.get("image_format", "png"),
            distilled_lora_strength=float(data.get("distilled_lora_strength", 0.6)),
            depth_lora_strength=float(data.get("depth_lora_strength", 0.6)),
            detailer_lora_strength=float(data.get("detailer_lora_strength", 0.48)),
            camera_static_lora_strength=float(data.get("camera_static_lora_strength", 0.7))
        )

def decode_base64_image(image_data: str, image_format: str = "png") -> Image.Image:
    """Decode base64 image data to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def upload_to_s3(file_path: str, s3_key: str, content_type: str = "video/mp4") -> str:
    """Upload file to S3 and return public URL"""
    try:
        s3_client.upload_file(
            file_path,
            AWS_S3_BUCKET,
            s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'ACL': 'public-read'
            }
        )
        
        # Generate public URL
        s3_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")

def download_models():
    """Download all required models (call this once at initialization)"""
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
    
    # Download LoRAs
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
    
    return model_paths, lora_paths

class VideoGenerator:
    """Singleton class to manage model loading and generation"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            if self.device == "cuda":
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU Memory: {total_mem:.1f} GB")
                self.use_fp8 = total_mem < 32
                if self.use_fp8:
                    print("Enabling FP8 mode")
            else:
                self.use_fp8 = False
            
            # Download models
            self.model_paths, self.lora_paths = download_models()
            self._initialized = True
    
    def generate_video(self, request: GenerationRequest, output_path: str) -> str:
        """Generate video based on request parameters"""
        
        # Calculate video configuration
        video_config = calculate_video_config(
            duration_seconds=request.duration_seconds,
            orientation=request.orientation,
            frame_rate=request.frame_rate,
            base_resolution=request.base_resolution,
            num_inference_steps=request.num_inference_steps,
            cfg_guidance_scale=request.video_cfg_scale,  # Use video CFG for config calc
            seed=request.seed,
        )
        
        # Prepare images list (for i2v) - FIXED: Must be list[tuple[str, int, float]]
        images: List[Tuple[str, int, float]] = []
        temp_img_dir = None
        
        if request.image_data:
            try:
                # Decode base64 to PIL Image
                image = decode_base64_image(request.image_data, request.image_format)
                
                # Create temp directory for image
                temp_img_dir = tempfile.mkdtemp()
                temp_img_path = os.path.join(temp_img_dir, f"input_image.{request.image_format}")
                
                # Save image to disk (pipeline requires file path)
                image.save(temp_img_path)
                
                # Format: (file_path, frame_index, strength)
                images = [(temp_img_path, 0, 1.0)]
                print(f"Using image input for i2v. Saved to: {temp_img_path}, Size: {image.size}")
            except Exception as e:
                print(f"Warning: Failed to decode image, falling back to t2v: {str(e)}")
                images = []
        
        try:
            # Configure tiling for high resolution
            print("Configuring VAE tiling for high resolution...")
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=320,
                    tile_overlap_in_pixels=64
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=64,
                    tile_overlap_in_frames=24
                )
            )
            
            # Calculate chunks for progress reporting
            video_chunks_number = get_video_chunks_number(
                video_config["num_frames"], tiling_config
            )
            
            print(f"""
            Generating {video_config['num_frames']} frames @ {video_config['frame_rate']}fps
            Resolution: {video_config['width']}x{video_config['height']}
            Duration: ~{video_config['num_frames'] / video_config['frame_rate']:.1f}s
            Inference steps: {video_config['num_inference_steps']}
            Chunks: {video_chunks_number}
            Mode: {'i2v' if images else 't2v'}
            
            VIDEO Guidance:
              CFG Scale: {request.video_cfg_scale}
              STG Scale: {request.video_stg_scale} (Blocks: {request.video_stg_blocks})
              A2V Scale: {request.video_modality_scale}
              Rescale: {request.video_rescale_scale}
            
            AUDIO Guidance (Optimized):
              CFG Scale: {request.audio_cfg_scale} (Higher for clarity)
              STG Scale: {request.audio_stg_scale} (Blocks: {request.audio_stg_blocks})
              V2A Scale: {request.audio_modality_scale} (For lipsync)
              Rescale: {request.audio_rescale_scale}
            """)
            
            # Initialize LoRAs with user-provided strengths
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=self.model_paths["distilled_lora_path"],
                    strength=request.distilled_lora_strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
            
            loras = [
                LoraPathStrengthAndSDOps(
                    path=self.lora_paths["depth"],
                    strength=request.depth_lora_strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ),
                LoraPathStrengthAndSDOps(
                    path=self.lora_paths["detailer"],
                    strength=request.detailer_lora_strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ),
                LoraPathStrengthAndSDOps(
                    path=self.lora_paths["camera_static"],
                    strength=request.camera_static_lora_strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ),
            ]
            
            # Initialize pipeline with current request parameters
            print("Initializing TI2VidTwoStagesPipeline...")
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=self.model_paths["checkpoint_path"],
                distilled_lora=distilled_lora,
                spatial_upsampler_path=self.model_paths["upsampler_path"],
                gemma_root=self.model_paths["gemma_root"],
                loras=loras,
                device=self.device,
                fp8transformer=self.use_fp8,
            )
            
            # Generate video with OPTIMIZED separate guider params for Video and Audio
            with torch.no_grad():
                video_iterator, audio = pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or "",
                    seed=video_config["seed"],
                    height=video_config["height"],
                    width=video_config["width"],
                    num_frames=video_config["num_frames"],
                    frame_rate=video_config["frame_rate"],
                    num_inference_steps=video_config["num_inference_steps"],
                    
                    # VIDEO guidance params
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=request.video_cfg_scale,
                        stg_scale=request.video_stg_scale,
                        rescale_scale=request.video_rescale_scale,
                        modality_scale=request.video_modality_scale,
                        skip_step=request.video_skip_step,
                        stg_blocks=request.video_stg_blocks if request.video_stg_blocks else [],
                    ),
                    
                    # AUDIO guidance params - OPTIMIZED defaults
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=request.audio_cfg_scale,  # 7.0 default for stronger prompt adherence
                        stg_scale=request.audio_stg_scale,  # 1.0 default for temporal coherence
                        rescale_scale=request.audio_rescale_scale,
                        modality_scale=request.audio_modality_scale,  # 3.0 default for V2A lipsync
                        skip_step=request.audio_skip_step,
                        stg_blocks=request.audio_stg_blocks if request.audio_stg_blocks else [],
                    ),
                    
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=request.enhance_prompt,
                )
                
                print(f"Encoding video to {output_path}...")
                encode_video(
                    video=video_iterator,
                    fps=video_config["frame_rate"],
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )
            
            # Clean up pipeline
            del pipeline
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"✓ Video saved to {output_path}")
            return output_path
            
        finally:
            # Clean up temp image directory if it was created
            if temp_img_dir and os.path.exists(temp_img_dir):
                try:
                    shutil.rmtree(temp_img_dir)
                    print(f"Cleaned up temp image directory: {temp_img_dir}")
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp directory {temp_img_dir}: {e}")

# Lazy-initialized singleton for serverless usage
video_generator: Optional[VideoGenerator] = None

async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless
    Expected job format:
    {
        "input": { ... generation parameters ... }
    }
    """
    global video_generator
    temp_dir = None
    
    try:
        # Lazy initialize the heavy VideoGenerator on first request
        if video_generator is None:
            video_generator = VideoGenerator()

        # Parse input
        input_data = job.get("input", {})
        
        # Validate required parameters
        if not input_data.get("prompt"):
            return {
                "error": "Missing required parameter: 'prompt'",
                "success": False
            }
        
        # Validate frame rate
        frame_rate = float(input_data.get("frame_rate", 12.0))
        if frame_rate <= 0 or frame_rate > 60:
            return {
                "error": "Frame rate must be between 0.1 and 60 fps",
                "success": False
            }
        
        # Validate duration
        duration_seconds = float(input_data.get("duration_seconds", 25.0))
        if duration_seconds <= 0 or duration_seconds > 120:
            return {
                "error": "Duration must be between 1 and 120 seconds",
                "success": False
            }
        
        # Create request object
        request = GenerationRequest.from_dict(input_data)
        
        # Generate unique filename
        video_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp()
        local_video_path = os.path.join(temp_dir, f"{video_id}.mp4")
        s3_key = f"{S3_PREFIX}{video_id}.mp4"
        
        # Generate video
        print(f"Starting video generation for request ID: {video_id}")
        print(f"Parameters: {duration_seconds}s @ {frame_rate}fps, {request.orientation}, {request.base_resolution}p")
        
        output_path = video_generator.generate_video(request, local_video_path)
        
        # Upload to S3
        print(f"Uploading video to S3: {s3_key}")
        s3_url = upload_to_s3(output_path, s3_key)
        
        # Return result
        return {
            "success": True,
            "video_url": s3_url,
            "video_id": video_id,
            "message": "Video generated successfully and uploaded to S3",
            "details": {
                "duration_seconds": request.duration_seconds,
                "frame_rate": request.frame_rate,
                "orientation": request.orientation,
                "resolution": f"{request.base_resolution}p",
                "mode": "i2v" if request.image_data else "t2v",
                "inference_steps": request.num_inference_steps,
                "video_cfg_scale": request.video_cfg_scale,
                "audio_cfg_scale": request.audio_cfg_scale,
                "audio_stg_enabled": request.audio_stg_scale > 0,
                "v2a_scale": request.audio_modality_scale
            }
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in handler: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Clean up on error
        if video_generator is not None and video_generator.device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "error": str(e),
            "traceback": error_trace,
            "success": False
        }
        
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")

def run_sync_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for async handler (for RunPod compatibility)"""
    return asyncio.run(handler(job))

# RunPod Serverless Entry Point
runpod.serverless.start({"handler": handler})