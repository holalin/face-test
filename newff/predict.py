import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import joblib
import numpy as np
import cv2  # 用于显示图片
import os

# ================= 配置与路径 =================
model_dir = "./model"
classifier_path = "emotion_classifier.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的 7 种情绪映射表（数据集标签：1惊讶 2恐惧 3厌恶 4开心 5伤心 6愤怒 7平淡）
emotion_labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Angry",
    7: "Neutral",
}

# ================= 1. 加载模型 =================
print("正在加载 DINOv3 和分类器...")
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir).to(device)
model.eval()
clf = joblib.load(classifier_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = frame[y : y + h, x : x + w]
    else:
        roi = frame

    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    inputs = processor(images=img_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    probs = clf.predict_proba(feat)[0]
    return probs


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
    else:
        print("按 q 退出实时表情识别")
        history_probs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            probs = predict_from_frame(frame)
            history_probs.append(probs)
            if len(history_probs) > 10:
                history_probs.pop(0)

            avg_probs = np.mean(history_probs, axis=0)
            best_index = int(np.argmax(avg_probs))
            real_label_id = clf.classes_[best_index]
            confidence = avg_probs[best_index] * 100
            emotion_name = emotion_labels.get(real_label_id, f"ID: {real_label_id}")

            prob_str = []
            for cls_id, p in zip(clf.classes_, probs):
                name = emotion_labels.get(cls_id, f"ID: {cls_id}")
                prob_str.append(f"{name}: {p * 100:.1f}%")
            print(" | ".join(prob_str))
            text = f"{emotion_name} {confidence:.1f}%"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Emotion Recognition - DINOv3 (Webcam)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
