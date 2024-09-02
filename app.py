import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from NeuralNet import NeuralNet

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Configurations
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10

# Load the trained model (Assuming you have a trained model saved as 'model.pth')
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('model/model.pt', map_location=device))
model.to(device)
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Gradio function to process the image and make predictions
def predict(image):
    # Load the image
    image = Image.fromarray(image)

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)
    image = image.view(-1, 28*28)  # Flatten the image

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return int(predicted.item())

# Create a Gradio interface
interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(), 
                         outputs="label", 
                         live=False, 
                         title="Digit Recognizer using Feed-Forward Nueral Network",
                         description="Upload a digit image to recognize it")

# Launch the interface
if __name__ == "__main__":
    interface.launch()
