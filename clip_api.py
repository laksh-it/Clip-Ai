import os
import threading
import requests
import torch
import clip
import logging
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
# Allow only localhost:3000 and your backend Render URL (update with your real URL)
CORS(app, origins=["http://localhost:3000", "https://your-render-url.onrender.com"])

# Set up rate limiting (5 requests per minute per IP)
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])

# Logging for monitoring traffic
logging.basicConfig(level=logging.INFO)

# Health-check endpoint (so your ping returns 200)
@app.route("/")
def home():
    return "Alive", 200

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Expanded category list
category_list = [
    "Cars", "Automobiles", "Sports", "Movies", "TV", "Music", "Entertainment",
    "Celebrities", "Pop Culture", "Cities", "Architecture", "Minimalist", "Aesthetic",
    "Technology", "Cyber", "Gaming", "Anime", "Manga", "Fantasy", "Mythology", "Gods", "Religion",
    "Science", "Exploration", "Nature", "Landscapes", "Space", "Astronomy", "History", "Heritage",
    "Books", "Literature", "Philosophy", "Thought", "Food", "Culinary", "Fashion", "Trends",
    "Photography", "Visual Arts", "Military", "Warfare", "Bikes", "Motorcycles", "Fitness", "Health",
    "Travel", "Adventure", "Horror", "Dark Themes", "Superheroes", "Comics", "Sci-Fi", "Futuristic",
    "Medicine", "Healthcare", "Racing", "Wildlife", "Animals", "Mythical Creatures",
    "Luxury", "Lifestyle", "DIY", "Crafting", "Education", "Learning", "Psychology", "Human Behavior",
    "Anime Culture", "Gaming Culture", "Vehicles", "Transportation", "War", "Battles",
    "Luxury Homes", "Real Estate", "Vintage", "Retro", "Hobbies", "Interests", "Occult", "Mysticism",
    "Martial Arts", "Combat", "Cybersecurity", "Hacking", "Spirituality", "Mindfulness", "Folklore", "Legends",
    "Economic World", "Financial World", "Academia", "Research", "Paranormal", "Supernatural",
    "Comedy", "Humor", "Historical Monuments", "Cinema Culture", "Fitness Challenges",
    "Space Missions", "Rockets", "Art", "Museums", "Extreme Sports", "Dark Web", "Conspiracies",
    "Digital Art", "NFTs", "Astrology", "Zodiac Signs", "Luxury Watches", "Accessories",
    "Car Shows", "Exhibitions", "Environmental Awareness", "Robotics", "AI", "Camping", "Survival",
    "Political Science", "Governance", "Streetwear", "Urban Culture", "Theme Parks", "Attractions",
    "Motivation", "Self-Help", "Languages", "Cultures", "Animal Conservation", "Zoos",
    "Gaming Consoles", "Tech", "Psychological Thrillers", "Motorcycle Stunts", "Racing",
    "Puzzles", "Riddles", "Festivals", "Celebrations", "Classic Cars", "Historical Cars", "Geography",
    "Extreme Weather", "Natural Disasters", "Phobias", "Irrational Fears", "Military Vehicles", "Tanks",
    "Survival Stories", "Vintage Films", "Luxury Hotels", "Resorts", "Ancient History",
    "Lost Civilizations", "Science Experiments", "Discoveries", "Supercars", "Tech Evolution",
    "Wildlife Photography", "Street Photography", "Urban Life", "Exotic Animals", "Yachts", "Marine Life",
    "Artificial Intelligence", "Space Exploration", "The Unknown", "Survival Tactics", "Strategies",
    "Online Communities", "Fandoms", "Tattoo Art", "Body Modification", "Classic Cartoons", "Animation",
    "Urban Legends", "Myths", 

    # **New Nature-Based Categories**
    "Mountains", "Forests", "Rivers", "Waterfalls", "Beaches", "Oceans",
    "Clouds", "Skies", "Moon", "Stars", "Sunsets", "Sunrises",
    "Storms", "Lightning", "Snow", "Ice", "Rain", "Thunder",
    "Lakes", "Ponds", "Deserts", "Dunes", "Greenery", "Fields"
]

@limiter.limit("5 per minute")
@app.route('/classify', methods=['POST'])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files["image"]
    if not image.filename.lower().endswith(("png", "jpg", "jpeg")):
        return jsonify({"error": "Invalid file type"}), 400
    
    image_path = "temp.jpg"
    image.save(image_path)

    categories = get_best_categories(image_path)
    
    return jsonify({"categories": categories})

def get_best_categories(image_path):
    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**image_inputs)

    text_inputs = processor(text=category_list, padding=True, truncation=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)

    similarity = torch.matmul(image_features, text_features.T)
    best_indices = similarity.topk(5).indices.tolist()[0]  

    return [category_list[i] for i in best_indices]

# Updated Keep-Alive Function (no while loop; recursive timer)
def keep_alive():
    PING_URL = "https://your-render-url.onrender.com"  # Replace with your actual Render URL
    try:
        response = requests.get(PING_URL)
        print(f"✅ Pinged Render! Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Ping failed: {e}")
    # Increase the time interval here; 600 seconds = 10 minutes
    threading.Timer(600, keep_alive).start()

# Start the Keep-Alive Thread
threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
