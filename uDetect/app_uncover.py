import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os


"""

this script is for peeled potato detection using YOLOv8 model

"""

# Load the YOLO model
model = YOLO("uDetect/uncovered_potato.pt")

def predict_potato(image):
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    results = model.predict(source=image, show=False, conf=0.6)
    
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Create Gradio interface
iface = gr.Interface(
    fn=predict_potato,
    inputs=gr.Image(type="pil"),  
    outputs=gr.Image(type="pil"),
    title="Potato Defect Detection with YOLO11",
    description="Upload an image of a potato to detect if it's covered or uncovered using a trained YOLO11 model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)