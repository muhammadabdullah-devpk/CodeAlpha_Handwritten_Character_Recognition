import numpy as np
import cv2
from PIL import Image


def _preprocess_core(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    # 1. Convert image to grayscale
    img_gray = image.convert("L")
    img_np = np.array(img_gray)
    
    # 10. Automatically detect background color using border pixels
    h, w = img_np.shape
    border_pixels = np.concatenate([
        img_np[0, :],
        img_np[-1, :],
        img_np[:, 0],
        img_np[:, -1]
    ])
    avg_border = np.mean(border_pixels)
    
    # MNIST expects a white digit (255) on a black background (0).
    # If the border average is bright, it indicates a white background, so we invert.
    if avg_border > 127:
        img_np = cv2.bitwise_not(img_np)
        
    # 2. Detect the handwritten digit using contour detection
    # Threshold the image to make it binary (white digit on black background)
    _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny contours (noise)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    if not valid_contours:
        valid_contours = contours
        
    if valid_contours:
        # Combined bounding box of all contours to preserve all parts of the digit
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for c in valid_contours:
            x, y, cw, ch = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)
            
        # 3 & 4. Crop only the digit and remove unnecessary whitespace
        if x_max > x_min and y_max > y_min:
            digit_crop = img_np[y_min:y_max, x_min:x_max]
        else:
            digit_crop = img_np
    else:
        digit_crop = img_np
        
    # 5. Preserve aspect ratio & 6. Add padding & 7. Center inside a square canvas
    crop_h, crop_w = digit_crop.shape
    max_side = max(crop_h, crop_w)
    
    # Compute padding (15% of max_side, minimum 4 pixels)
    padding = int(max_side * 0.15)
    if padding < 4:
        padding = 4
        
    canvas_size = max_side + 2 * padding
    
    # Create black square canvas
    square_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # Center the cropped digit on the canvas
    offset_x = padding + (max_side - crop_w) // 2
    offset_y = padding + (max_side - crop_h) // 2
    square_canvas[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w] = digit_crop
    
    # 8. Resize to exactly 28x28 pixels
    resized = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 9. Normalize pixel values to [0, 1]
    normalized = resized.astype("float32") / 255.0
    
    # 11. Prepare model input by reshaping to (1, 28, 28, 1)
    model_input = normalized.reshape(1, 28, 28, 1)
    
    # Preprocessed preview image (Pillow image)
    preview_image = Image.fromarray(resized)
    
    return model_input, preview_image


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    model_input, _ = _preprocess_core(image)
    return model_input


def prepare_preview_image(image: Image.Image) -> Image.Image:
    _, preview_image = _preprocess_core(image)
    return preview_image


def preprocess_uploaded_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    model_input, _ = _preprocess_core(image)
    return model_input
