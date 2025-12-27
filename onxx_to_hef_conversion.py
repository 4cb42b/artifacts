# Hailo v3.33, TF v2.18
import os; os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import random
import numpy as np
import cv2
from hailo_sdk_client import ClientRunner

model_name  = "yolo11m-pose"
onnx_path   = "yolo11m-pose.onnx"
hef_path    = "yolo11m-pose.hef"
images_path = "datasets/val2017" # Coco-pose val part, ~5k images
calib_count = 1024 # 1k images from 5k for optimization finetuning

# Transfrom data to optimizer-acceptable format
def get_calibration_data():
    all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_path}")
        
    subset_size = min(len(all_images), calib_count)
    selected_images = random.sample(all_images, subset_size)
    print(f"Loading {subset_size} images...")

    batch_list = []
    
    for filename in selected_images:
        filepath = os.path.join(images_path, filename)
        img = cv2.imread(filepath)
        if img is None: continue
            
        # 1. resize to net input
        img = cv2.resize(img, (640, 640))
        # 2. BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. convert to optimizer-acceptable format
        img = img.astype(np.float32)
        
        batch_list.append(img)
    
    # concatenate into one giant array [1024, 3, 640, 640]
    mega_array = np.stack(batch_list, axis=0)
    
    print(f"Calibration data shape: {mega_array.shape}")
    
    return {'yolo11m-pose/input_layer1': mega_array}

# Compile the model to har (intermediate format)

runner = ClientRunner(hw_arch='hailo8')

print(f"Parsing {onnx_path}...")
runner.translate_onnx_model(
    onnx_path, 
    model_name,
    start_node_names=['images'],
    end_node_names=[
        '/model.16/cv2/conv/Conv', 
        '/model.19/cv2/conv/Conv', 
        '/model.22/cv2/conv/Conv'
    ],
    net_input_shapes={'images': [1, 3, 640, 640]}
)

# add auto-conversion layer on Halio to save CPU
print("Applying Normalization Script...")
alls_script = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
"""
runner.load_model_script(alls_script)

# Optimization part

print("Running optimization (Quantization)...")

calib_data = get_calibration_data()

runner.optimize(calib_data, data_type="np_array")

print("Compiling to HEF...")
hef = runner.compile()

# Save
with open(hef_path, "wb") as f:
    f.write(hef)

print(f"Model saved to {hef_path}")
