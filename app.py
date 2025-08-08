from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import cv2
import io

app = Flask(__name__)

# Sarcastic replies based on plant mood
SASS = {
    'thriving': [
        "Yeah, I'm basically photosynthesizing greatness — try not to be jealous.",
        "Stop the applause, I'm doing all the work here (and sunlight, I guess)."
    ],
    'content': [
        "I could use a little attention, but sure — post a pic instead.",
        "Happy enough. Keep the compliments coming; they fertilize me emotionally."
    ],
    'thirsty': [
        "Water? Maybe? Or keep asking me how I'm doing, that's apparently my cardio.",
        "Thirst level: medium. Dramatic rescues not required."
    ],
    'sick': [
        "I'm vibing... toward the compost bin.",
        "Wow, were you trying to make me look rustic or just forgetful?"
    ],
    'sunburnt': [
        "Chill with the tanning bed, I'm not trying to audition for a desert documentary.",
        "Please stop, I'm sunburnt and emotionally sensitive."
    ]
}

def analyze_image_pxls(img_bytes):
    # Load image with PIL and convert to BGR for OpenCV
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = np.array(pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize to max 600px to speed up processing
    h, w = img.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    total_pixels = img.shape[0] * img.shape[1]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green mask
    lower_green = np.array([36, 40, 20])
    upper_green = np.array([86, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_count = cv2.countNonZero(green_mask)
    green_ratio = green_count / (total_pixels + 1e-6)

    # Yellow / brown mask
    lower_yellow = np.array([10, 40, 20])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_count = cv2.countNonZero(yellow_mask)
    yellow_ratio = yellow_count / (total_pixels + 1e-6)

    # Bright mask (overexposed)
    v_channel = hsv[:, :, 2]
    bright_ratio = np.sum(v_channel > 220) / (total_pixels + 1e-6)

    # Decide mood based on heuristics
    if green_ratio > 0.35:
        mood = 'thriving'
    elif green_ratio > 0.18 and yellow_ratio < 0.20:
        mood = 'content'
    elif yellow_ratio > 0.18 and green_ratio < 0.25:
        mood = 'thirsty'
    elif bright_ratio > 0.25 and green_ratio < 0.2:
        mood = 'sunburnt'
    else:
        mood = 'sick'

    return {
        'mood': mood,
        'green_ratio': round(float(green_ratio), 4),
        'yellow_ratio': round(float(yellow_ratio), 4),
        'bright_ratio': round(float(bright_ratio), 4)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    f = request.files['image']
    img_bytes = f.read()
    try:
        info = analyze_image_pxls(img_bytes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    mood = info['mood']
    message = np.random.choice(SASS.get(mood, SASS['sick']))

    return jsonify({
        'mood': mood,
        'message': message,
        'debug': info
    })

if __name__ == '__main__':
    app.run(debug=True)
