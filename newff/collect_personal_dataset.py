import cv2
import os
import time


def main():
    save_root = "dataset_personal"
    label_keys = {
        ord("1"): (1, "Surprise"),
        ord("2"): (2, "Fear"),
        ord("3"): (3, "Disgust"),
        ord("4"): (4, "Happy"),
        ord("5"): (5, "Sad"),
        ord("6"): (6, "Angry"),
        ord("7"): (7, "Neutral"),
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    current_label = None
    current_name = ""
    counters = {i: 0 for i in range(1, 8)}

    print("按 1-7 选择当前要采集的情绪标签，c 或空格拍照保存，按 q 退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        display = frame.copy()
        info = "1-7选择情绪, c/space拍照, q退出"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if current_label is None:
            status = "当前情绪: 未选择"
        else:
            status = f"当前情绪: {current_label}-{current_name}, 已采集 {counters[current_label]}"
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Collect Personal Dataset", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in label_keys:
            current_label, current_name = label_keys[key]
            print(f"切换到: {current_label}-{current_name}")
            continue

        if key in (ord("c"), 32):
            if current_label is None:
                print("请先按 1-7 选择情绪标签")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi = frame[y : y + h, x : x + w]
            else:
                roi = frame

            label_dir = os.path.join(save_root, "train", str(current_label))
            os.makedirs(label_dir, exist_ok=True)
            counters[current_label] += 1
            filename = f"{int(time.time())}_{counters[current_label]}.jpg"
            path = os.path.join(label_dir, filename)
            cv2.imwrite(path, roi)
            print(f"已保存: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

