import base64
import requests
from openai import OpenAI
import cv2
import numpy as np
import os


class ImageAnalyzer:
    def __init__(self):
        """
        Initializes the ImageAnalyzer with OpenAI API credentials.
        """
        self.client = OpenAI()

    def encode_image(self, image):
        """
        Encodes an image to Base64.

        Parameters:
        image (numpy array): Image array (BGR format).

        Returns:
        str: Base64 encoded string of the image.
        """
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def analyze_images(self, images, detail="low"):
        """
        Takes a list of cropped images and returns descriptions from OpenAI's Vision model.

        Parameters:
        images (list of numpy arrays): Cropped images from the original frame.
        detail (str): "low", "high", or "auto" to set image processing detail.

        Returns:
        list of str: Descriptions of objects in each image.
        """
        image_data = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img)}", "detail": detail},
            }
            for img in images
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is in these images? Answer with only nouns and/or adjective, only give one answer, if you are unsure, say unknown"}] + image_data,
                }
            ],
            max_tokens=300,
        )

        # Accessing the content of the response using dot notation
        return [choice.message.content for choice in response.choices]


def main():
    # Define the directory containing cropped images
    cropped_images_dir = "cropped_images"
    if not os.path.exists(cropped_images_dir):
        print(f"Directory '{cropped_images_dir}' does not exist.")
        return
    
    # Load all images from the directory
    images = []
    for filename in os.listdir(cropped_images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(cropped_images_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Could not read image {image_path}")
    
    if not images:
        print("No valid images found in the directory.")
        return
    
    # Initialize ImageAnalyzer
    analyzer = ImageAnalyzer()
    
    # Analyze images
    results = analyzer.analyze_images(images)
    
    print(results[0])

if __name__ == "__main__":
    main()