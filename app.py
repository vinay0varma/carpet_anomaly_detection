import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
from PIL import Image
import torch.nn as nn

# Import your external model
from feat_cae import FeatCAE

# Allow loading custom class safely
torch.serialization.add_safe_globals({'FeatCAE': FeatCAE})

# Define ResNet-style Autoencoder
class ResNetFeatureExtractorAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Feature extractor with projection to 1536 channels
class FeatureExtractorWithProjection(nn.Module):
    def __init__(self, output_channels=1536):
        super().__init__()
        self.backbone = efficientnet_b4(pretrained=True).features
        self.project = nn.Conv2d(1792, output_channels, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        projected = self.project(features)
        return projected

# Model loader
def load_model(model_type, model_path):
    if model_type == "FeatCAE":
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        model = ResNetFeatureExtractorAutoencoder()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image for input and feature extraction
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((380, 380)),  # EfficientNet B4 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 3, 380, 380]

    extractor = FeatureExtractorWithProjection()
    features_1536 = extractor(tensor)  # [1, 1536, H, W]

    return tensor, features_1536

# Calculate anomaly score
def calculate_anomaly_score(original, reconstructed):
    error = torch.mean((original - reconstructed) ** 2).item()
    return error

# Streamlit App
def main():
    st.title("Anomaly Detection in carpets")

    model_type = st.selectbox("Select Model Type", ["FeatCAE", "ResNetFeatureExtractorAutoencoder"])
    model_path = st.text_input(r"C:\Users\poorn\OneDrive\ドキュメント\anomaly_detection\autoencoder_with_resnet_deep_features1.pth")
    
    if model_path:
        model = load_model(model_type, model_path)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            input_tensor, features_1536 = preprocess_image(image)

            with torch.no_grad():
                if model_type == "FeatCAE":
                    input_for_model = features_1536
                else:
                    input_for_model = input_tensor

                reconstructed = model(input_for_model)

            anomaly_score = calculate_anomaly_score(input_for_model, reconstructed)
            st.write(f"**Anomaly Score:** {anomaly_score:.6f}")

            THRESHOLD = 0.5
            if anomaly_score > THRESHOLD:
                st.error("Anomaly Detected!")
            else:
                st.success("Image is Normal.")

if __name__ == "__main__":
    main()
