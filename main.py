import pickle
from datetime import datetime

import io
import base64

# import face_recognition
from PIL import Image

import torch
from flask import Flask, render_template, request, jsonify

from edith.main import Language, evaluate
from edith.models import EncoderRNN, AttnDecoderRNN
from torchvision import transforms

import numpy as np

# from facial.emotions.models import MyModel
from facial.pain.models import get_convnext

app = Flask(__name__)

# EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']
PAINS = ['pain', 'no pain']
conversation_history = []
detected_emotion = ""
detected_identity = ""


def predict(X_img, knn_clf, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    # Load image file and find face locations


#    X_face_locations = face_recognition.face_locations(X_img)

# If no faces are found in the image, return an empty result.
#    if len(X_face_locations) == 0:
#        return []

# Find encodings for faces in the test iamge
#    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

# Use the KNN model to find the best matches for the test face
#    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
#    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

# Predict classes and remove classifications that aren't within the threshold
#    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
#            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


pain_conv = [
    "Bonjour <NAME>, je suis Tiago, un robot d'accompagnement, et je suis là pour satisfaire vos besoins. Comment allez-vous aujourd'hui ?",
    "Bonjour Tiago, je vais bien merci",
    "Il me semble que vous souffrez de douleurs",
    "Oui, depuis hier soir",
    "Vous avez mal ou exactement ?",
    "au bas du dos",
    "Ok, je vais informer votre médecin",
    "Merci Tiago."
]

i = 0


@app.route('/message', methods=['POST'])
def message():
    msg = str(request.data).split("'")[1]
    '''if msg == '<$>':
        conversation_history.clear()
        if detected_emotion == "happy":
            resp = "Bonjour <NAME>, comment vous sentez-vous aujourd’hui ? Je vois que vous êtes content."
        elif detected_emotion == "sad":
            resp = "Bonjour <NAME>, comment vous sentez-vous aujourd'hui? J’ai l’impression que vous avez l’air triste."
        else:
            resp = "Bonjour <NAME>, comment vous sentez-vous aujourd'hui?"
    else:
        conversation_history.append(msg)
        test_output = evaluate(encoder, decoder, language, " SEP ".join(conversation_history), max_length, device)[0]
        del test_output[-1]
        resp = language.words_to_sentence(test_output)'''

    global i
    resp = pain_conv[i]
    i += 2

    resp = resp.replace("<NAME>", detected_identity) \
        .replace("<TIME>", datetime.now().strftime("%I:%M"))
    conversation_history.append(resp)

    return jsonify(resp)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    img_part = str(request.data).split(',')[1]
    # Convert the binary data to a PIL image
    pil_image = Image.open(io.BytesIO(base64.b64decode(img_part))).convert('RGB')
    pil_image = pil_image.resize((640, 480))

    # Find all people in the image using a trained classifier model
    # Note: You can pass in either a classifier file name or a classifier model instance
    #    predictions = predict(np.array(pil_image), knn_clf)

    persons = []
    rects = []
    #    for name, (top, right, bottom, left) in predictions:
    #        persons.append(name)
    #        rects.append([left, top, right - left, bottom - top])

    global detected_identity
    #    detected_identity = name

    '''emotions = np.array([])
    if len(rects) != 0:
        top, right, bottom, left = predictions[0][1]
        cropped_face = pil_image.crop((left, top, right, bottom))

        transform = transforms.Compose([transforms.Resize(48),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])

        input_faces = transform(cropped_face).unsqueeze(0)
        emotions = np.array(EMOTIONS)[np.argmax(model(input_faces).detach().numpy(), axis=1)]
        global detected_emotion
        detected_emotion = emotions[0]'''

    # Directement appliquer le modèle de détection de la douleur à l'image entière
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_image = pil_image.convert("RGB")  # Convertir en RGB si nécessaire
    input_image = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output, _ = model(input_image)

    prediction = torch.argmax(output[0], dim=0).item()
    pain = "DOULEURS" if prediction == 1 else "PAS DE DOULEURS"

    global detected_pain
    detected_pain = pain

    return jsonify({'rects': rects, 'pain': pain})


@app.route("/")
def home():
    return render_template('index.html')


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used: " + device)

    # Load emotion recognition model
    '''model = MyModel(5, training=False)  # For recognition
    state_dict = torch.load('facial/emotions/pretrained/best_model.pkl', map_location=device)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()'''

    # Définissez votre modèle
    model = get_convnext(num_classes=2, model_size='tiny')

    # Spécifiez le chemin vers le fichier .pt contenant les poids du modèle
    checkpoint_path = "facial/pain/pretrained/model.pt"

    # Chargez les poids du modèle à partir du fichier .pt
    #    checkpoint = torch.load(checkpoint_path, map_location=device)
    #    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load face recognition model
    model_path = "facial/identification/pretrained/trained_knn_model.clf"

    #    with open(model_path, 'rb') as f:
    #        knn_clf = pickle.load(f)

    # Load language models
    hidden_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #    package = torch.load('edith/out/checkpoints/best_model_v1.pkl', map_location=device)
    #    max_length = package['max_length']
    #    language = package['language']

    #    encoder = EncoderRNN(language.n_words, hidden_size, device=device).to(device)
    #    decoder = AttnDecoderRNN(hidden_size, language.n_words, device, max_length, dropout_p=0.1).to(device)
    #    encoder.load_state_dict(package['encoder_state_dict'])
    #    decoder.load_state_dict(package['decoder_state_dict'])
    #    encoder.eval()
    #    decoder.eval()
    app.run()
