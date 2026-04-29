# 🎬 LTX-2 RunPod Serverless Endpoint

Welcome to the **LTX-2 RunPod Serverless** deployment repository! This codebase provides a fully containerized, serverless API endpoint for generating high-quality videos using the [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) video generation model.

This project is tailored for deployment on **RunPod Serverless**, offering an efficient, auto-scaling backend that turns text and images into stunning cinematic videos. It includes built-in AWS S3 integrations, advanced memory handling (FP8 mode/VAE Tiling), and support for multiple LoRAs (Depth, Detailer, Camera Control).

---

## 🚀 Why Serverless Endpoints on RunPod?

Deploying this intensive video generation model as a Serverless Endpoint on RunPod comes with massive advantages:

1. **💸 Cost-Effective (Pay-per-second):** Video generation models require expensive GPUs (like A100s or H100s). With serverless, you only pay for the exact seconds the GPU is generating a video. When the API is idle, you pay **$0** for compute.
2. **📈 Infinite Auto-Scaling:** Whether you receive 1 request a day or 1,000 requests a minute, RunPod scales the number of active GPU workers automatically to handle the traffic queue, then scales back to zero.
3. **🛠️ Zero Infrastructure Management:** You don't have to worry about server maintenance, OS updates, handling crashed instances, or load balancing. You simply deploy the Docker container and let RunPod route the API requests.
4. **⚡ Blazing Fast Cold Starts:** By utilizing RunPod's optimized network volume caching, cold starts are minimized, getting your workers ready for inference quickly.

---

## 📁 Codebase Architecture & File Purposes

Here is an in-depth breakdown of every file in this repository and its exact purpose:

### `Dockerfile`
The blueprint for the container environment. 
- Uses a CUDA 12.8 Ubuntu base image.
- Installs all system-level dependencies for video processing (like `ffmpeg`).
- Integrates `uv` (a blazing fast Python package manager) to install dependencies significantly faster than pip.
- Clones the official LTX-2 repository and installs its local packages.
- Caches huggingface models and sets up the serverless handler (`handler.py`) as the container's entry point.

### `handler.py`
The brain of the RunPod API. 
- **API Interface:** Listens to incoming JSON payloads from RunPod containing generation parameters (prompt, frame rate, resolution, guidance scales).
- **Singleton Model Manager:** Initializes the `VideoGenerator` class dynamically on the first request to load models into VRAM efficiently.
- **Image-to-Video (I2V):** Handles base64 image decoding if a starting image is provided.
- **S3 Integration:** Automatically uploads the resulting `.mp4` video to an AWS S3 bucket and returns a publicly accessible URL to the end-user.

### `inference.py`
The core AI engine. 
- Contains the advanced mathematical and configuration logic required to run the LTX-2 `TI2VidTwoStagesPipeline`.
- **Resolution & Tiling:** Configures VAE (Variational Autoencoder) spatial and temporal tiling limits. This is a critical feature that prevents the GPU from running out of memory (OOM errors) or throwing 32-bit math limits when generating high-resolution (1080p) videos.
- **LoRA Integration:** Pre-downloads and merges specialized Lightricks LoRAs (`Depth-Control`, `Detailer`, `Camera-Control-Static`) into the diffusion pipeline to enhance visual quality and adherence.

### `requirements.txt`
The precise list of Python dependencies.
- Heavily optimized for the PyTorch CUDA 12.8 ecosystem.
- Includes strict version locking for RunPod SDK, Transformers, Hugging Face Hub, and video manipulation libraries (like `av`, `imageio-ffmpeg`).

### `.env`
Your local environment variables file.
- Currently houses your Hugging Face Token (`HF_TOKEN`) needed to bypass gate-restricted models or weights. *Note: Ensure this token has read access to the specific LTX-2 models.*

### `.dockerignore`
Prevents unnecessary or sensitive files from being copied into the Docker image.
- Ensures `.env` files (containing secrets) and local `.log` files do not bloat the container or compromise security.

---

## 🛠️ How to Deploy & Use

### 1. Build and Push the Docker Image
First, build your Docker container and push it to a container registry (like Docker Hub or GitHub Container Registry):
```bash
# Build the image
docker build -t your-username/ltx2-runpod-serverless:latest .

# Push to your registry
docker push your-username/ltx2-runpod-serverless:latest
```

### 2. Configure RunPod Serverless
1. Go to your **RunPod Dashboard** -> **Serverless** -> **Templates** and create a new template.
2. Set the **Container Image** to `your-username/ltx2-runpod-serverless:latest`.
3. Set the **Container Registry Credentials** if your registry is private.
4. Set the **Environment Variables** (crucial for downloading models and uploading to S3):
   - `HF_TOKEN`: Your Hugging Face access token (must have read access to LTX-2 models).
   - `AWS_ACCESS_KEY_ID`: Your AWS access key.
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key.
   - `AWS_REGION`: Your AWS region (e.g., `us-east-1`).
   - `AWS_S3_BUCKET`: The name of the S3 bucket where you want videos to be uploaded.
5. Create a new **Serverless Endpoint** using this template. Select a GPU with at least 24GB of VRAM (e.g., RTX 3090, RTX 4090, A100, or H100).

### 3. Make an API Request
Once your endpoint is deployed, you can trigger a video generation request by sending a POST request with the required payload. 

Here is an example using `curl`:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
     -d '{
       "input": {
         "prompt": "A majestic golden retriever puppy playing joyfully in a sunlit meadow...",
         "duration_seconds": 8.0,
         "frame_rate": 24.0,
         "orientation": "landscape",
         "base_resolution": 1088,
         "num_inference_steps": 50,
         "video_cfg_scale": 4.5
       }
     }'
```

### 4. Typical Request Workflow
1. RunPod receives the request and spins up a worker.
2. `handler.py` catches the payload and initializes the `VideoGenerator` (downloading/loading weights into VRAM on the first cold start).
3. `inference.py` processes the VAE tiling and applies LoRAs to generate the frames.
4. The `.mp4` is exported locally to a temporary directory.
5. `handler.py` automatically uploads the file to your configured AWS S3 bucket.
6. The RunPod API responds with a JSON containing the public `video_url`!
