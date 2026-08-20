import os
import torch
import numpy as np
# 修正 1：DINOv3 使用 AutoImageProcessor 和 AutoModel
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# ================= 配置 =================
# 建议有显卡的话优先使用 "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================= 1️⃣ 加载 DINOv3 本地模型 =================
# 确保该目录下有：model.safetensors, config.json, preprocessor_config.json
model_dir = "./model" 

try:
    # 修正 2：AutoImageProcessor 会自动识别 preprocessor_config.json
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    # 修正 3：AutoModel 会自动加载 .safetensors 权重
    model = AutoModel.from_pretrained(model_dir, use_safetensors=True)
    model.to(device)
    model.eval()
    print("Successfully loaded DINOv3 model and processor.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure transformers library version >= 4.56.0 and model files are complete.")

# ================= 2️⃣ 图像预处理函数 =================
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    # 修正 4：最新的 Processor 调用接口
    inputs = image_processor(images=img, return_tensors="pt")
    return inputs['pixel_values'].to(device)

# ================= 3️⃣ 单张图像特征提取 =================
@torch.no_grad()
def extract_feature(img_path):
    pixel_values = preprocess(img_path)
    outputs = model(pixel_values)
    
    # ⚠️ 重要提示：DINOv3 的输出结构
    # last_hidden_state 的形状通常是 [batch_size, 1 + 4 + N_patches, hidden_dim]
    # 其中 0 是 CLS token (全局特征)
    # 1-4 是 Register tokens (寄存器令牌，用于消除伪影)
    # 5 往后才是 Patch tokens (局部特征)
    
    # 获取 CLS token 特征用于情绪分类和不一致性分析
    feat = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
    return feat.cpu().numpy().squeeze()

# ================= 4️⃣ 遍历数据集 =================
def process_split(split_path):
    features, labels = [], []
    if not os.path.exists(split_path):
        print(f"Path not found: {split_path}")
        return np.array([]), np.array([])
        
    for label_name in sorted(os.listdir(split_path)):
        label_dir = os.path.join(split_path, label_name)
        if not os.path.isdir(label_dir):
            continue
        try:
            label = int(label_name)
        except ValueError:
            continue
            
        print(f"Processing label {label} from {label_name} ...")
        for img_name in os.listdir(label_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(label_dir, img_name)
            try:
                feat = extract_feature(img_path)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"Skip image {img_name}: {e}")
                
    return np.array(features), np.array(labels)

# ================= 5️⃣ 主运行逻辑 =================
if __name__ == "__main__":
    train_dir = "dataset_personal/train"
    test_dir  = "dataset_personal/test"

    if os.path.exists(train_dir):
        print("Extracting TRAIN features...")
        X_train, y_train = process_split(train_dir)
        if X_train.size > 0:
            np.save("features_train.npy", X_train)
            np.save("labels_train.npy", y_train)
            print(f"Saved Train: {X_train.shape}")

    if os.path.exists(test_dir):
        print("Extracting TEST features...")
        X_test, y_test = process_split(test_dir)
        if X_test.size > 0:
            np.save("features_test.npy", X_test)
            np.save("labels_test.npy", y_test)
            print(f"Saved Test: {X_test.shape}")

    print("Extraction Done!")
