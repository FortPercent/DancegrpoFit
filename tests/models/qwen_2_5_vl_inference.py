import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.nn import functional as F

# ==== 1. 本地加载模型 ====
model_path = "/root/Qwen2.5-VL-7B-Instruct"  # 修改为你本地的权重路径
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# ==== 2. 随机文本 ====
text_input = "This is a randomly generated sentence."
text_inputs = tokenizer(text_input, return_tensors="pt").to(device)

with torch.no_grad():
    text_embeds = model.model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        output_hidden_states=True,
        return_dict=True
    ).last_hidden_state  # [batch, seq_len, hidden]
    # 取 CLS 位置或平均池化
    text_features = text_embeds.mean(dim=1)

# ==== 3. 随机视频 ====
# 模拟视频 pixel_values: (batch, num_frames, 3, H, W)
pixel_values = torch.rand(1, 8, 3, 224, 224, device=device, dtype=torch.float16)

with torch.no_grad():
    # Qwen2.5-VL 的视觉编码器通常是 model.model.visual
    video_embeds = model.visual(pixel_values, torch.tensor([[1, 14, 14]], device=device))
    video_features = video_embeds.mean(dim=0, keepdim=True)  # 平均池化成 [1, hidden]

# ==== 4. 计算余弦相似度 ====
text_features = F.normalize(text_features, p=2, dim=-1)
video_features = F.normalize(video_features, p=2, dim=-1)

cosine_sim = (text_features @ video_features.T).item()
print(f"Cosine similarity between text and video features: {cosine_sim:.4f}")