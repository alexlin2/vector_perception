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

    def analyze_images(self, images, detail="auto"):
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
                    "content": [{"type": "text", "text": "What is in these images? Give a short word answer with at most two words, if not sure, say unknown"}] + image_data,
                }
            ],
            max_tokens=300,
            timeout=5,
        )

        # Accessing the content of the response using dot notation
        return [choice.message.content for choice in response.choices][0]


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
    
    # Split results into a list of items
    object_list = [item.strip()[2:] for item in results.split('\n')]

    # Overlay text on images and display them
    for i, (img, obj) in enumerate(zip(images, object_list)):
        if obj:  # Only process non-empty lines
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            text = obj.strip()
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text at top of image
            x = 10
            y = text_height + 10
            
            # Add white background for text
            cv2.rectangle(img, (x-5, y-text_height-5), (x+text_width+5, y+5), (255,255,255), -1)
            # Add text
            cv2.putText(img, text, (x, y), font, font_scale, (0,0,0), thickness)
            
            # Save or display the image
            cv2.imwrite(f"annotated_image_{i}.jpg", img)
            print(f"Detected object: {obj}")


if __name__ == "__main__":
    main()