# LTX-Video 2.3 口播视频生成 + LatentSync 口型同步

**在 16GB 显存上实现完整的 AI 口播视频生成管线：角色一致性 + 音频驱动口型同步**

从零到生产级的 AI 口播视频方案，突破了多个 16GB VRAM 的硬性限制。

## 效果展示

输入：一张角色照片 + 一段播客音频  
输出：角色口型与音频精确同步的口播视频

| 阶段 | 效果 |
|---|---|
| LTX-Video 生成 | 角色说话、表情自然、首尾帧一致 |
| LatentSync 1.6 后处理 | 口型与音频精确同步、512px 无融合痕迹 |

## 技术亮点

### 1. 16GB 显存跑 22B 视频模型
- LTX-Video 2.3 22B 模型通过 nvfp4 量化 + DynamicVRAM 管理在 16GB 上推理
- 完整的音视频 AV 管线（编码、采样、分离、解码）

### 2. LatentSync 1.6 在 16GB 上运行（官方要求 18GB）
- **UNet CPU Offload**：推理完后 `self.unet.to("cpu")` 释放 ~4GB 显存
- **VAE 逐帧 CPU 解码**：避免 4GB 连续内存分配，利用 64GB 系统内存
- **Windows WDDM 12.74GB 限制突破**：通过精确的 GPU↔CPU 内存调度

### 3. ComfyUI 兼容性修复
- NestedTensor `clone()` 方法缺失修复
- LTXVSeparateAVLatent 普通 tensor fallback
- LatentSync GBK 编码兼容修复

### 4. 自定义角色 LoRA 训练
- RunComfy H100 云端训练（$5, 2000 步）
- 38 张训练图（真实照片 + Flux AI 生成）
- 触发词驱动的角色一致性

## 系统要求

| 组件 | 最低要求 |
|---|---|
| GPU | NVIDIA 16GB+ (测试: RTX 5070 Ti) |
| RAM | 32GB+，推荐 64GB（LatentSync CPU 解码用）|
| OS | Windows 11 |
| ComfyUI | v0.19.3+ |
| Python | 3.12+ |
| CUDA | 12.0+（编译需 12.8+）|

## 安装

### 1. ComfyUI 自定义节点

```bash
cd ComfyUI/custom_nodes

# LTX Podcast 节点（分镜循环）
git clone <this-repo> ComfyUI-LTX-Podcast

# MelBandRoFormer（人声分离）
git clone https://github.com/kijai/ComfyUI-MelBandRoFormer.git

# LatentSync（口型同步）
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
pip install -r ComfyUI-LatentSyncWrapper/requirements.txt
pip install decord DeepCache pytorch_lightning
```

### 2. 下载模型

**LTX-Video 2.3：**
- 主模型: `ltx-2.3-22b-dev-nvfp4.safetensors` → `models/checkpoints/`
- 文本编码器: `gemma_3_12B_it_fp4_mixed.safetensors` → `models/text_encoders/`
- 蒸馏 LoRA: `ltx-2.3-22b-distilled-lora-resized_dynamic_rank_159_fro09_bf16.safetensors` → `models/loras/LTX/`

**LatentSync 1.6：**
```bash
# UNet (5GB)
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints/

# VAE
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir checkpoints/vae/
```

### 3. 应用补丁

**ComfyUI NestedTensor 修复** (`comfy/nested_tensor.py`):
```python
# 在 _copy 方法后添加:
def clone(self):
    return NestedTensor([t.clone() for t in self.tensors])
```

**LTXVSeparateAVLatent 修复** (`comfy_extras/nodes_lt.py`):
```python
# execute 方法中，添加非 NestedTensor 处理:
samples = av_latent["samples"]
is_nested = getattr(samples, "is_nested", False)
if is_nested:
    latents = samples.unbind()
else:
    latents = samples.unbind()
    if len(latents) < 2:
        latents = (samples, torch.zeros_like(samples[:, :, :1]))
```

**LatentSync CPU Offload** (`latentsync/pipelines/lipsync_pipeline.py`):
```python
# decode_latents 方法替换为逐帧 CPU 解码:
def decode_latents(self, latents):
    latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    vae_device = next(self.vae.parameters()).device
    vae_dtype = next(self.vae.parameters()).dtype
    self.vae.to("cpu").float()
    torch.cuda.empty_cache()
    import gc; gc.collect()
    latents_cpu = latents.to("cpu").float()
    decoded_frames = []
    for i in range(latents_cpu.shape[0]):
        frame = self.vae.decode(latents_cpu[i:i+1]).sample
        decoded_frames.append(frame)
    decoded_latents = torch.cat(decoded_frames, dim=0)
    del decoded_frames, latents_cpu; gc.collect()
    self.vae.to(vae_device, dtype=vae_dtype)
    return decoded_latents.to(latents.device)

# __call__ 方法中，采样循环后添加 UNet offload:
self.unet.to("cpu")
torch.cuda.empty_cache()
# ... decode_latents ...
synced_video_frames.append(decoded_latents.cpu())
self.unet.to(device)
```

## 使用方法

### 一键生成每日播客视频

```bash
python daily_podcast_video.py 2026-04-22
```

自动完成：
1. 启动 ComfyUI → 分段生成口播视频（8秒/段）→ ffmpeg 拼接
2. 关闭 ComfyUI → 启动 LatentSync 1.6 → 512px 口型同步
3. 输出最终视频

### 单独运行各阶段

```bash
# 仅生成视频（不做口型同步）
python podcast_video_gen.py 2026-04-22 3 8042  # 日期 段数 种子

# 仅做口型同步（输入已有视频）
python run_lipsync_standalone.py

# 参数调优测试
python run_best.py
python run_batch_v2.py
python run_final_tuning.py
```

## 最优参数

### LTX-Video 生成

| 参数 | 值 | 说明 |
|---|---|---|
| 角色 LoRA 强度 | 0.85 | 低于 0.8 不够像，高于 1.0 有伪影 |
| 蒸馏 LoRA 强度 | 0.3 | 加速采样，去掉会模糊 |
| 首尾帧引导 | 0.7 | 原始工作流值，0.95 过强限制运动 |
| 采样步数 | 15 | 8 步太少，15 步性价比最高 |
| 分辨率 | 1024x576 | 横版口播最佳 |
| 种子 | 8042 | 口型最好的种子 |

### LatentSync 1.6

| 参数 | 值 | 说明 |
|---|---|---|
| 配置 | stage2_512.yaml | 512px 无融合痕迹 |
| 推理步数 | 30 | stage2_512 默认 |
| guidance_scale | 2.0 | 口型表现力 |
| 批量 | 1 | 最小化显存峰值 |

### 提示词

**正面（slim 版防胖脸）：**
```
xixi, a young Chinese woman, 22 years old, bangs and long dark brown hair with hair clip, 
delicate slim oval face, large bright brown eyes, small nose, youthful appearance, fair skin, 
sitting at a cafe table, wearing a cozy cream cardigan over white top, speaking to camera, 
mouth moving while talking, warm cafe lighting with plants, close-up portrait, natural soft expression
```

**负面：**
```
blurry, out of focus, overexposed, underexposed, noise, grainy, poor lighting, flickering, 
motion blur, distorted, unnatural skin, dark, grey, dull, watermark, text, logo, subtitle, 
fat face, chubby, wide face, puffy
```

## 架构

```
┌─────────────────────────────────────────────────┐
│                 daily_podcast_video.py           │
├──────────────────────┬──────────────────────────┤
│   Phase 1: LTX-Video │   Phase 2: LatentSync    │
│                      │                          │
│  podcast.wav         │  raw_video.mp4           │
│       ↓              │       ↓                  │
│  TrimAudio(8s)       │  LoadVideo(193 frames)   │
│       ↓              │       ↓                  │
│  AudioVAEEncode      │  Whisper(audio→embed)    │
│       ↓              │       ↓                  │
│  FirstFrame Guide    │  UNet(16frame batches)   │
│  + LastFrame Guide   │   [GPU → CPU offload]    │
│       ↓              │       ↓                  │
│  AV Concat+NoiseMask │  VAE Decode              │
│       ↓              │   [CPU, frame-by-frame]  │
│  CropGuides          │       ↓                  │
│       ↓              │  RestoreVideo(paste back) │
│  Euler 15-step       │       ↓                  │
│       ↓              │  lipsync_result.mp4      │
│  SeparateAV+Decode   │                          │
│       ↓              │                          │
│  raw_video.mp4       │                          │
└──────────────────────┴──────────────────────────┘
```

## LoRA 训练

如需训练自己的角色 LoRA：

1. 准备 30-50 张角色照片（统一分辨率，无水印）
2. 为每张写英文 caption（以触发词开头）
3. 上传到 [RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/ltx-2-lora-training)
4. 参数：LTX-2.3, rank 32, lr 1e-4, 2000-3000 步
5. 下载 .safetensors 放到 ComfyUI loras 目录

**注意：** 
- 避免混入带水印的图片（会学到水印伪影）
- AI 生成的辅助图建议用目标模型（LTX）而非其他模型（Flux）生成
- 16GB 显存不够本地训练 22B 模型

## 踩坑记录

| 问题 | 原因 | 解决 |
|---|---|---|
| ComfyUI LTXVCropGuides 崩溃 | NestedTensor 缺 clone() | 补丁 nested_tensor.py |
| LTXVSeparateAVLatent 失败 | 采样器破坏 NestedTensor 结构 | 添加普通 tensor fallback |
| LoRA 训练 Windows 失败 | transformers/PyTorch 兼容性 | WSL2 + CUDA 12.8 |
| LoRA 训练 OOM | 16GB 不够 22B 模型 | RunComfy 云端 ($5) |
| 人物不像角色 | Flux AI 图稀释真实面部特征 | slim 提示词 + 调参 |
| 视频人脸偏胖 | LTX 模型偏差 | negative prompt: fat face |
| LatentSync OOM | WDDM 12.74GB 限制 | UNet CPU offload |
| LatentSync 融合伪影 | 256px 分辨率太低 | 升级到 1.6 + stage2_512 |
| LatentSync 512px CPU OOM | 4GB 连续分配失败 | 逐帧 VAE 解码 |

## 致谢

- [LTX-Video 2.3](https://github.com/Lightricks/LTX-2) by Lightricks
- [LatentSync 1.6](https://github.com/bytedance/LatentSync) by ByteDance
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper)
- [ComfyUI-MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer)
- [RunComfy](https://www.runcomfy.com) for LoRA training

## License

MIT
