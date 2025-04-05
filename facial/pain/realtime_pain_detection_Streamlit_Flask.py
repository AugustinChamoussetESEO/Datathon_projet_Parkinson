import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from models import get_convnext

# Définissez votre modèle
model = get_convnext(num_classes=2)

# Spécifiez le chemin vers le fichier .pt contenant les poids du modèle
checkpoint_path = "out/model.pt"

# Chargez les poids du modèle à partir du fichier .pt
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fonction pour dessiner l'étiquette sur l'image
def draw_label(image, label):
    font = ImageFont.truetype("arial.ttf", 20)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)

# Définir l'en-tête de l'application
st.title("Détection en temps réel de la douleur")

# Initialiser la webcam
cap = cv2.VideoCapture(0)

# Boucle de capture vidéo en temps réel
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image OpenCV en image PIL
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prétraitement de l'image
    input_image = transform(frame_pil).unsqueeze(0)

    # Effectuer l'inférence
    with torch.no_grad():
        output, _ = model(input_image)

    # Obtenez la prédiction
    prediction = torch.argmax(output[0], dim=0).item()
    label = "Douleur" if prediction == 1 else "Pas de douleur"

    # Dessiner l'étiquette sur l'image
    draw_label(frame_pil, label)

    # Convertir l'image PIL en image OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Afficher l'image avec l'étiquette
    st.image(frame, channels="BGR")

# Libérer la webcam
cap.release()
