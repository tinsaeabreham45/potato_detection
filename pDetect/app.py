import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def detect_and_annotate_potato(image, model_path="pDetect/my_model.pt", confidence_threshold=0.4):

    model = YOLO(model_path)


    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    results = model.predict(source=image_bgr, conf=confidence_threshold, show=False)


    class_colors = {
        'Damaged Potato': (0, 0, 255),
        'Defected Potato': (255, 0, 0),
        'Potato': (0, 255, 0),
        'Sprouted Potato': (255, 255, 0)
    }


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness_font = 5
    thickness = 2


    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = model.names[cls_id]
            label = f"{label_name} ({conf:.2f})"
            color = class_colors.get(label_name, (255, 255, 255))
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_bgr, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(image_bgr, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness_font)


    resized_image = cv2.resize(image_bgr, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)


    output_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(output_image)

# Create Gradio interface
demo = gr.Interface(
    fn=detect_and_annotate_potato,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Detected Potatoes"),
    title="Potato Detection with YOLO",
    description="Upload an image to detect and annotate potatoes using a YOLO model.",
    examples=["images/test1.jpg"]
)


if __name__ == "__main__":
    demo.launch(share=True)