from flask import Flask, request, jsonify, abort
from functools import wraps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Secret key for accessing the endpoint
SECRET_KEY = 'kokojiji'

try:
   
    model = load_model('./models/productValidation.h5')
    print(model.summary())  # Mencetak ringkasan model untuk memastikan model dimuat dengan benar
except Exception as e:
    print(f"Error loading the model: {e}")


def pad_and_resize_image(image, target_size=(64, 64)):
    """
    Add padding to the input image to make it square and resize to the target size.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_width, img_height = image.size
    if img_width > img_height:
        delta_width = 0
        delta_height = img_width - img_height
    else:
        delta_width = img_height - img_width
        delta_height = 0
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    padded_image = ImageOps.expand(image, padding)
    resized_image = padded_image.resize(target_size)
    return resized_image

def prepare_image(image, target_size=(64, 64)):
    """
    Prepare the image for model prediction.
    """
    image = pad_and_resize_image(image, target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

MAX_CLASS_PROBABILITY_THRESHOLD = 0.8
MIN_BBOX_AREA_THRESHOLD = 0.3

def is_image_rejected(class_probabilities, bounding_box, image_size=(224, 224)):
    """
    Determine if an image is accepted based on class probabilities and bounding box predictions.
    """
    # Check class probabilities
    max_class_prob = np.max(class_probabilities)
    if max_class_prob < MAX_CLASS_PROBABILITY_THRESHOLD:
        return False # No Harmful object detected

    # Check bounding box size
    xmin, ymin, xmax, ymax = bounding_box
    bbox_area = (xmax - xmin) * (ymax - ymin)
    img_area = image_size[0] * image_size[1]

    # print(bbox_area, img_area)
    # Check if bounding box area is within acceptable range
    if MIN_BBOX_AREA_THRESHOLD * img_area <= bbox_area:
        return True  # Harmful object detected

    return False

def require_secret_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-SECRET-KEY') != SECRET_KEY:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return jsonify({'success': False, 'message': 'API under construction', 'data': []})

@app.route('/predict', methods=['POST'])
@require_secret_key
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file)
        image = prepare_image(image, target_size=(224, 224))
        predictions = model.predict(image)
        predictions_list = [np.array(pred) for pred in predictions]
        # Get the max value from the second list of predictions
        max_prob = float(max(predictions[1][0]))
        bbox = predictions_list[0][0]*224

        # Determine acceptance or rejection
        if is_image_rejected(predictions_list[1][0], bbox):
            decision = 'reject'
        else:
            decision = 'accept'

        # Convert predictions to list for JSON serialization
        predictions_list_serializable = [pred.tolist() for pred in predictions_list]

        return jsonify({'prediction': predictions_list_serializable, 'max_probability': max_prob, 'decision': decision})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'success': False, 'message': 'Anda tersesat ayo kembali ke jalan yang benar', 'data': []}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
