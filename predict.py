import argparse
import numpy as np

from src.model_builder import load_saved_model
from src.preprocessing import preprocess_uploaded_image


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from image")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    model = load_saved_model()
    processed = preprocess_uploaded_image(args.image)
    probs = model.predict(processed, verbose=0)[0]
    prediction = int(np.argmax(probs))

    print(f"Predicted digit: {prediction}")
    print("Top confidence scores:")
    for digit, prob in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {digit}: {prob:.4f}")


if __name__ == "__main__":
    main()
