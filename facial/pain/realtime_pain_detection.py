import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from models import get_convnext


# Définissez votre modèle
model = get_convnext(num_classes=2, model_size='tiny')

# Spécifiez le chemin vers le fichier .pt contenant les poids du modèle
checkpoint_path = "pretrained/model.pt"

# Chargez les poids du modèle à partir du fichier .pt
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()



# Initialiser la caméra
cap = cv2.VideoCapture(0)

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
    draw.text((10, 10), label, fill=(255, 0, 0), font=font)  # Couleur rouge

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
        output, _ = model(input_image)  # Modification ici pour gérer la sortie du modèle

    # Obtenez la prédiction
    prediction = torch.argmax(output[0], dim=0).item()  # Modification ici pour extraire les prédictions
    label = "pain" if prediction == 1 else "no pain"

    # Dessiner l'étiquette sur l'image
    draw_label(frame_pil, label)

    # Convertir l'image PIL en image OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Afficher l'image avec l'étiquette
    cv2.imshow('Pain Detection', frame)

    # Quitter la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libérer la caméra et détruire toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
