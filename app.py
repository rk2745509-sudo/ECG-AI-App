import streamlit as st
import numpy as np
import cv2
import pickle
from pywt import wavedec
from skimage.feature import hog
import matplotlib.pyplot as plt
from datetime import datetime
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ECG AI Diagnosis | Premium",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

/* Animated background - FIXED ATTACHMENT TO PREVENT SCROLL JUMPING */
.stApp {
    background: linear-gradient(-45deg, #070913, #111528, #1a2138, #070913);
    background-size: 400% 400%;
    background-attachment: fixed; 
    animation: gradientShift 15s ease infinite;
    color: white;
}
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

/* Title */
.title {
    font-size: 52px; font-weight: bold; text-align: center;
    background: linear-gradient(135deg, #00E5FF, #2962FF, #00E5FF);
    -webkit-background-clip: text; color: transparent;
    margin-bottom: 40px;
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
}

/* Base Premium Card */
.premium-card {
    background: rgba(20, 25, 40, 0.55);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 30px;
    border: 1px solid rgba(0, 229, 255, 0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

/* ---------- THE PERFECT COMMON SIZE FIX FOR GUIDE CARDS ---------- */
.split-card {
    background: rgba(20, 25, 40, 0.6);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(0, 229, 255, 0.15);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    transition: transform 0.3s ease, border-color 0.3s ease;
    height: 260px; /* THIS FORCES ALL 4 CARDS TO BE EXACTLY THE SAME SIZE! */
    display: flex;
    flex-direction: column;
}
.split-card:hover {
    transform: translateY(-8px);
    border-color: #00E5FF;
    box-shadow: 0 15px 35px rgba(0, 229, 255, 0.2);
}
.split-card-num {
    font-size: 2.8rem;
    font-weight: 900;
    color: #00E5FF;
    margin-bottom: 5px;
    text-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
}
.split-card-title {
    font-size: 1.3rem;
    color: white;
    font-weight: bold;
    margin-bottom: 12px;
}
.split-card-desc {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ---------- Uploader Responsive Ultimate Fix ---------- */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(10, 15, 30, 0.95) !important; 
    border: 2px dashed #00E5FF !important; 
    border-radius: 15px !important;
    padding: 20px 10px !important;
}

/* Force elements to stack vertically */
[data-testid="stFileUploadDropzone"] > div > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 12px !important;
}

[data-testid="stFileUploadDropzone"] * { 
    color: #ffffff !important; 
}

/* Hide the "Limit 200MB..." text to save space when sidebar opens */
[data-testid="stFileUploadDropzone"] small {
    display: none !important; 
}

/* Make text smaller and centered */
[data-testid="stFileUploadDropzone"] span {
    font-size: 15px !important;
    text-align: center !important;
}

/* Fix the button size and position */
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(90deg, #2962FF, #00E5FF) !important; 
    color: white !important;
    border-radius: 10px !important; 
    border: none !important; 
    font-weight: bold !important;
    padding: 8px 20px !important;
    width: 100% !important; /* Adapts to small width perfectly */
    margin-top: 5px !important;
}

/* Preview Image */
.zoom-cb { display: none; }
.zoom-img { width: 100%; border-radius: 12px; cursor: zoom-in; transition: transform 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
.zoom-cb:checked + label .zoom-img {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    object-fit: contain; background: rgba(0,0,0,0.9); z-index: 99999;
    cursor: zoom-out; border-radius: 0; padding: 20px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #00E5FF, #2962FF) !important;
    border-radius: 40px !important; color: #ffffff !important; font-weight: bold !important; width: 100% !important; margin-top: 15px;
}

/* Expanders - Overlap Fix */
[data-testid="stExpander"] {
    border: none !important; /* Removes Streamlit's default outer border */
    margin-top: 15px;
}
[data-testid="stExpander"] details {
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    border-radius: 15px !important;
    background: rgba(10, 15, 30, 0.6) !important;
}
[data-testid="stExpander"] details summary {
    background: rgba(20, 25, 40, 0.9) !important;
    border: none !important; /* Removes inner border */
    border-radius: 15px !important;
    padding: 15px 20px !important;
    font-weight: bold !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(7, 9, 19, 0.95) !important; border-right: 1px solid rgba(0, 229, 255, 0.2); }
h1, h2, h3, h4, p, label, li, span, .stMarkdown { color: #ffffff !important; }

/* Result UI */
.result-card-normal {
    background: rgba(0, 229, 255, 0.08); border-left: 5px solid #00E5FF; padding: 20px; border-radius: 12px; display: flex; align-items: center; margin-bottom: 25px; margin-top: 10px;
}
.result-card-abnormal {
    background: rgba(255, 51, 102, 0.08); border-left: 5px solid #FF3366; padding: 20px; border-radius: 12px; display: flex; align-items: center; margin-bottom: 25px; margin-top: 10px;
}
.result-text { margin-left: 20px; }
.result-title { font-size: 26px; font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: 1px; }

/* Progress Bars */
.prob-container { margin-bottom: 15px; }
.prob-label-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-weight: 600; font-size: 0.9rem; color: #e2e8f0; text-transform: uppercase; }
.prob-track { width: 100%; height: 10px; background: rgba(0,0,0,0.6); border-radius: 10px; }
.prob-fill-normal { height: 100%; background: linear-gradient(90deg, #2962FF, #00E5FF); border-radius: 10px; box-shadow: 0 0 10px rgba(0,229,255,0.6); }
.prob-fill-abnormal { height: 100%; background: linear-gradient(90deg, #FF9800, #FF3366); border-radius: 10px; box-shadow: 0 0 10px rgba(255,51,102,0.6); }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        with open("ecg_model_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file `ecg_model_artifacts.pkl` not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

artifacts = load_model()
model = artifacts["model"]
scaler = artifacts["scaler"]
pca = artifacts["pca"]
encoder = artifacts["encoder"]
class_names = artifacts["class_names"]

# ---------- HELPER FUNCTIONS ----------
@st.cache_data(ttl=3600)
def preprocess_image_bytes(uploaded_file_bytes):
    file_bytes = np.asarray(bytearray(uploaded_file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("Could not decode image.")
    img = cv2.resize(img, (256, 256))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

@st.cache_data(ttl=3600)
def extract_features_cached(img):
    coeffs = wavedec(img, 'db1', level=3)
    dwt_features = np.concatenate([c.flatten() for c in coeffs])[:100]
    hog_features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2))
    stats = [np.mean(img), np.std(img), np.median(img)]
    return np.concatenate([dwt_features, hog_features, stats])

def occlusion_heatmap(img, model, scaler, pca, encoder, step=16):
    h, w = img.shape
    original_features = extract_features_cached(img)
    original_features_scaled = scaler.transform(original_features.reshape(1, -1))
    original_features_pca = pca.transform(original_features_scaled)
    original_pred = model.predict(original_features_pca)[0]
    original_prob = model.predict_proba(original_features_pca)[0][original_pred]

    heatmap = np.zeros((h, w), dtype=np.float32)
    step = max(step, 32)
    for y in range(0, h, step):
        for x in range(0, w, step):
            img_copy = img.copy()
            img_copy[y:y+step, x:x+step] = 128
            try:
                occ_features = extract_features_cached(img_copy)
                occ_features_pca = pca.transform(scaler.transform(occ_features.reshape(1, -1)))
                occ_pred = model.predict(occ_features_pca)[0]
                if occ_pred == original_pred: change = 0
                else:
                    occ_prob = model.predict_proba(occ_features_pca)[0][original_pred]
                    change = max(0, original_prob - occ_prob)
                heatmap[y:y+step, x:x+step] = change
            except: heatmap[y:y+step, x:x+step] = 0
    if heatmap.max() > 0: heatmap = heatmap / heatmap.max()
    return heatmap

def generate_report(predicted_class, confidence, probabilities, filename):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"ECG AI Diagnosis Premium Report\n--------------------------------\nDate: {timestamp}\nImage: {filename}\n\nPrediction: {predicted_class}\nConfidence: {confidence:.4f}\n\nClass Probabilities:\n"
    for cls, prob in zip(class_names, probabilities):
        report += f"  {cls}: {prob:.4f}\n"
    report += "\nDisclaimer:\nThis is an AI-based prediction tool for research purposes only."
    return report

# ---------- HTML COMPONENT: HOLLOW 3D HEART ----------
def get_ecg_heart_component(predicted_class, confidence):
    confidence_val = float(confidence)
    escaped_class = predicted_class.replace('"', '\\"')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; font-family: 'Segoe UI', sans-serif; color: white; overflow: hidden; }}
            .dashboard {{
                background: rgba(10, 15, 30, 0.7); backdrop-filter: blur(20px); border-radius: 2rem;
                padding: 1.5rem; width: 100%; border: 1px solid rgba(0, 229, 255, 0.2);
                display: flex; flex-direction: column; align-items: center;
            }}
            .heart-container {{ position: relative; width: 100%; max-width: 250px; height: 220px; display: flex; justify-content: center; align-items: center; margin-bottom: 0.5rem; }}
            canvas#wireframeHeart {{ width: 100%; height: 100%; }}
            
            .ecg-container {{ background: rgba(0, 0, 0, 0.5); border-radius: 1rem; padding: 0.5rem; width: 100%; border: 1px solid rgba(255, 255, 255, 0.1); display: flex; justify-content: center; }}
            canvas#ecgLine {{ display: block; width: 100%; height: 60px; }}
            
            .status-box {{ text-align: center; margin-top: 1rem; background: rgba(0, 0, 0, 0.4); padding: 0.75rem 1.5rem; border-radius: 2rem; border: 1px solid rgba(255,255,255,0.05); }}
            .diagnosis {{ font-size: 1.3rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }}
            .normal-text {{ background: linear-gradient(135deg, #00E5FF, #2962FF); -webkit-background-clip: text; color: transparent; }}
            .abnormal-text {{ background: linear-gradient(135deg, #FF3366, #FF9800); -webkit-background-clip: text; color: transparent; }}
            .confidence {{ font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="heart-container">
                <canvas id="wireframeHeart" width="250" height="220"></canvas>
            </div>
            <div class="ecg-container">
                <canvas id="ecgLine" width="250" height="60"></canvas>
            </div>
            <div class="status-box">
                <div class="diagnosis" id="diagText">Initializing...</div>
                <div class="confidence" id="confText">Analyzing core vitals</div>
            </div>
        </div>

        <script>
            const aiClass = "{escaped_class}";
            const aiConf = {confidence_val};
            const isNormal = aiClass.toLowerCase() === 'normal';
            
            const config = {{
                bpm: isNormal ? 72 : (aiConf > 0.8 ? 140 : 110),
                irregular: !isNormal,
                diagnosis: isNormal ? 'NORMAL' : 'ABNORMAL',
                cssClass: isNormal ? 'normal-text' : 'abnormal-text'
            }};

            document.getElementById('diagText').innerText = config.diagnosis;
            document.getElementById('diagText').className = 'diagnosis ' + config.cssClass;
            document.getElementById('confText').innerText = isNormal ? 'System Stable' : 'Confidence: ' + (aiConf * 100).toFixed(1) + '%';

            const canvasH = document.getElementById('wireframeHeart');
            const ctxH = canvasH.getContext('2d');
            
            const particles = [];
            const numParticles = 220; 
            const baseScale = 5.0; 
            
            for(let i=0; i<numParticles; i++) {{
                let t = Math.random() * Math.PI * 2;
                let rScale = Math.pow(Math.random(), 0.5); 
                let x = 16 * Math.pow(Math.sin(t), 3) * baseScale;
                let y = -(13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t)) * baseScale;
                
                particles.push({{
                    x: x * rScale, y: y * rScale, phase: Math.random() * Math.PI * 2,
                    ratio: Math.max(0, Math.min(1, (x * rScale + 70) / 140))
                }});
            }}

            function drawDynamicHeart(timestamp) {{
                let time = timestamp / 1000;
                ctxH.clearRect(0, 0, canvasH.width, canvasH.height);
                
                const centerX = canvasH.width / 2;
                const centerY = canvasH.height / 2 - 5; 
                
                let pulse = 1.0;
                let phase = (time % (60/config.bpm)) / (60/config.bpm);
                if (phase < 0.15) pulse = 1.1;
                else if (phase > 0.25 && phase < 0.4) pulse = 1.05;
                if (config.irregular) pulse += (Math.random() - 0.5) * 0.03;

                ctxH.save();
                ctxH.translate(centerX, centerY);
                ctxH.scale(pulse, pulse);

                let grad = ctxH.createLinearGradient(-70, 0, 70, 0);
                if(isNormal) {{
                    grad.addColorStop(0, 'rgba(0, 229, 255, 0.15)');
                    grad.addColorStop(1, 'rgba(255, 51, 102, 0.15)');
                }} else {{
                    grad.addColorStop(0, 'rgba(255, 51, 102, 0.2)');
                    grad.addColorStop(1, 'rgba(255, 100, 0, 0.2)');
                }}
                
                ctxH.beginPath();
                for (let t = 0; t <= Math.PI * 2; t += 0.1) {{
                    let px = 16 * Math.pow(Math.sin(t), 3) * baseScale;
                    let py = -(13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t)) * baseScale;
                    if (t === 0) ctxH.moveTo(px, py); else ctxH.lineTo(px, py);
                }}
                ctxH.fillStyle = grad;
                ctxH.fill();

                ctxH.globalCompositeOperation = 'screen';
                
                for(let i=0; i<particles.length; i++) {{
                    let p = particles[i];
                    let px = p.x + Math.sin(time*2 + p.phase)*2;
                    let py = p.y + Math.cos(time*2 + p.phase)*2;
                    
                    let pointRatio = isNormal ? p.ratio : 1.0;
                    let cR = Math.floor(0 * (1-pointRatio) + 255 * pointRatio);
                    let cG = Math.floor(229 * (1-pointRatio) + 51 * pointRatio);
                    let cB = Math.floor(255 * (1-pointRatio) + 102 * pointRatio);
                    
                    ctxH.fillStyle = 'rgba(' + cR + ',' + cG + ',' + cB + ',0.7)';
                    ctxH.fillRect(px-1.5, py-1.5, 3, 3);
                    
                    let connected = 0;
                    for(let j=i+1; j<particles.length && connected<3; j++) {{
                        let p2 = particles[j];
                        let p2x = p2.x + Math.sin(time*2 + p2.phase)*2;
                        let p2y = p2.y + Math.cos(time*2 + p2.phase)*2;
                        let distSq = (px-p2x)**2 + (py-p2y)**2;
                        
                        if(distSq < 500) {{ 
                            connected++;
                            let alpha = 0.3 - (distSq / 1666);
                            ctxH.strokeStyle = 'rgba(' + cR + ',' + cG + ',' + cB + ',' + alpha + ')';
                            ctxH.beginPath();
                            ctxH.moveTo(px, py);
                            ctxH.lineTo(p2x, p2y);
                            ctxH.stroke();
                        }}
                    }}
                }}
                ctxH.restore();
                requestAnimationFrame(drawDynamicHeart);
            }}
            requestAnimationFrame(drawDynamicHeart);

            const canvasE = document.getElementById('ecgLine');
            const ctxE = canvasE.getContext('2d');
            
            let ecgBuffer = new Array(canvasE.width).fill(canvasE.height / 2);
            let ecgPhase = 0;
            
            const gradE = ctxE.createLinearGradient(0, 0, canvasE.width, 0);
            if(isNormal) {{
                gradE.addColorStop(0, '#00E5FF');
                gradE.addColorStop(1, '#FF3366');
            }} else {{
                gradE.addColorStop(0, '#FF3366');
                gradE.addColorStop(1, '#FF5500');
            }}

            function drawECG() {{
                let delta = 1 / 60; 
                const beatInterval = 60 / config.bpm; 
                ecgPhase = (ecgPhase + delta / beatInterval) % 1;
                
                let y = 0;
                if (ecgPhase < 0.1) y = 20 * Math.sin((ecgPhase / 0.1) * Math.PI); 
                else if (ecgPhase > 0.15 && ecgPhase < 0.25) y = -10 * Math.sin(((ecgPhase - 0.15) / 0.1) * Math.PI); 
                else if (ecgPhase > 0.6 && ecgPhase < 0.75) y = 8 * Math.sin(((ecgPhase - 0.6) / 0.15) * Math.PI); 
                else y = (Math.random() - 0.5) * (config.irregular ? 8 : 1.5); 

                ecgBuffer.push((canvasE.height / 2) - y);
                ecgBuffer.shift(); 

                ctxE.clearRect(0, 0, canvasE.width, canvasE.height);
                
                ctxE.beginPath();
                for (let i = 0; i < ecgBuffer.length; i++) {{
                    if(i===0) ctxE.moveTo(i, ecgBuffer[i]);
                    else ctxE.lineTo(i, ecgBuffer[i]);
                }}
                
                ctxE.strokeStyle = isNormal ? 'rgba(150, 50, 255, 0.4)' : 'rgba(255, 51, 102, 0.4)';
                ctxE.lineWidth = 4;
                ctxE.lineJoin = 'round';
                ctxE.stroke();
                
                ctxE.strokeStyle = gradE;
                ctxE.lineWidth = 2;
                ctxE.stroke();
                
                requestAnimationFrame(drawECG);
            }}
            requestAnimationFrame(drawECG);
        </script>
    </body>
    </html>
    """
    return html

# Initialize session state for reset logic
if 'predicted_class' not in st.session_state: st.session_state.predicted_class = 'Normal'
if 'confidence' not in st.session_state: st.session_state.confidence = 1.0
if 'last_upload' not in st.session_state: st.session_state.last_upload = None

# ---------- MAIN UI ----------
st.markdown('<div class="title">ECG AI Diagnosis</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

# ---------- LEFT COLUMN: UPLOADER & PREVIEW ----------
with col_left:
    # THE EMPTY DECORATIVE BOX
      st.markdown('<div style="height: 60px; background: rgba(20, 25, 40, 0.55); backdrop-filter: blur(20px); border-radius: 28px; border: 1px solid rgba(0, 229, 255, 0.15); box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 25px;"></div>', unsafe_allow_html=True)

      
      st.markdown("""<h3 style="text-align:center; color:#00E5FF; margin-top: 0; margin-bottom: 25px; font-size: 1.6rem;">📤 Upload ECG Image</h3>""", unsafe_allow_html=True)

      uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="uploader")

      if not uploaded_file:
        st.session_state.predicted_class = 'Normal'
        st.session_state.confidence = 1.0
        st.session_state.last_upload = None
        st.markdown('<div style="text-align:center; opacity:0.5; padding:15px; margin-top: 10px;">⬆️ Waiting for image upload...</div>', unsafe_allow_html=True)
      else:
        try:
            img_bytes = uploaded_file.getvalue()
            b64_img = base64.b64encode(img_bytes).decode()
            
            st.markdown(f"""
            <div style="margin-top: 20px; text-align: center;">
                <p style='opacity:0.7; margin-bottom:8px; font-size:14px;'>Click image to enlarge</p>
                <input type="checkbox" id="zoom-img" class="zoom-cb">
                <label for="zoom-img">
                    <img src="data:image/jpeg;base64,{b64_img}" class="zoom-img" alt="Uploaded ECG">
                </label>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Processing ECG..."):
                img_preprocessed = preprocess_image_bytes(img_bytes)

            st.session_state.last_upload = (img_bytes, img_preprocessed, uploaded_file.name)

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.last_upload = None

# ---------- RIGHT COLUMN: RESULTS OR ALERT ----------
with col_right:
    # THE EXACT SAME EMPTY DECORATIVE BOX
    st.markdown('<div style="height: 60px; background: rgba(20, 25, 40, 0.55); backdrop-filter: blur(20px); border-radius: 28px; border: 1px solid rgba(0, 229, 255, 0.15); box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 25px;"></div>', unsafe_allow_html=True)
   
    if uploaded_file and st.session_state.last_upload is not None:
        try:
            img_bytes, img_preprocessed, filename = st.session_state.last_upload

            with st.spinner("🧠 Extracting features..."):
                features = extract_features_cached(img_preprocessed)
                features_scaled = scaler.transform(features.reshape(1, -1))
                features_pca = pca.transform(features_scaled)

            with st.spinner("📈 Making prediction..."):
                prediction = model.predict(features_pca)[0]
                probs = model.predict_proba(features_pca)[0]

            predicted_class = encoder.inverse_transform([prediction])[0]
            confidence = np.max(probs)

            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

            st.markdown("<h3 style='color:#00E5FF; margin-top:0; margin-bottom: 5px; font-size: 1.6rem;'>🧠 AI Diagnosis Result</h3>", unsafe_allow_html=True)
            
            is_normal = predicted_class.lower() == 'normal'
            card_class = "result-card-normal" if is_normal else "result-card-abnormal"
            icon = "✅" if is_normal else "⚠️"
            color = "#00E5FF" if is_normal else "#FF3366"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 35px; text-shadow: 0 0 15px {color};">{icon}</div>
                <div class="result-text">
                    <p class="result-title" style="color: {color};">{predicted_class.upper()}</p>
                    <p class="result-sub">Confidence Score: <b style="color: white; font-size: 15px;">{confidence * 100:.1f}%</b></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h3 style='color: white; font-size: 1.3rem;'>📊 Probabilities</h3>", unsafe_allow_html=True)
            
            for cls_name, prob in zip(class_names, probs):
                prob_pct = prob * 100
                fill_class = "prob-fill-normal" if cls_name.lower() == 'normal' else "prob-fill-abnormal"
                
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-label-row">
                        <span>{cls_name.upper()}</span>
                        <span style="color: {'#00E5FF' if cls_name.lower() == 'normal' else '#FF3366'};">{prob_pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="{fill_class}" style="width: {prob_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 Explainability (Occlusion Heatmap)"):
                st.markdown("Highlights regions of the ECG that most influenced the AI's decision.")
                if st.button("Generate Heatmap", use_container_width=True):
                    with st.spinner("Computing sensitivity..."):
                        heatmap = occlusion_heatmap(img_preprocessed, model, scaler, pca, encoder)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        fig.patch.set_facecolor('#070913')
                        ax.set_facecolor('#070913')
                        ax.imshow(img_preprocessed, cmap='gray')
                        im = ax.imshow(heatmap, cmap='hot', alpha=0.6)
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            # --- Ithu thaan main fix! Download button veliya vanthuruchu ---
            report = generate_report(predicted_class, confidence, probs, filename)

            st.markdown("<br>", unsafe_allow_html=True) 
            b_col1, b_col2, b_col3 = st.columns([1, 2, 1])

            with b_col2:
                st.download_button(
                    label="📥 Download Full Report", 
                    data=report, 
                    file_name=f"ECG_Report_{predicted_class}.txt", 
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            
    else:
        # INVISIBLE SPACER
        st.markdown("<h3 style='text-align:center; color:transparent; margin-top:0; margin-bottom:25px; font-size: 1.6rem; user-select:none;'>Placeholder</h3>", unsafe_allow_html=True)
        
        # ALIGNED INFO BOX
        st.markdown("""
        <div style="margin-top: 10px; background: rgba(0, 229, 255, 0.08); border-left: 4px solid #00E5FF; padding: 25px 20px; border-radius: 12px; display: flex; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <span style="font-size: 28px; margin-right: 15px;">👈</span>
            <span style="color: white; font-size: 17.5px; font-weight: 500; letter-spacing: 0.5px;">Please upload an ECG image to generate a prediction.</span>
        </div>
        """, unsafe_allow_html=True)
# ---------- PREMIUM SPLIT QUICK START GUIDE ----------
st.markdown("<h3 style='text-align: center; color: white; margin-top: 40px; margin-bottom: 25px; font-size: 2rem;'>📋 Quick Start Guide</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="split-card">
        <div class="split-card-num">01</div>
        <div class="split-card-title">Upload</div>
        <div class="split-card-desc">Select a clear ECG strip image (JPG/PNG format).</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="split-card">
        <div class="split-card-num">02</div>
        <div class="split-card-title">Process</div>
        <div class="split-card-desc">The AI will automatically grayscale and filter the image.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="split-card">
        <div class="split-card-num">03</div>
        <div class="split-card-title">Analyze</div>
        <div class="split-card-desc">Features (DWT & HOG) are extracted to match patterns.</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="split-card">
        <div class="split-card-num">04</div>
        <div class="split-card-title">Result</div>
        <div class="split-card-desc">View the diagnosis, confidence score, and download the report.</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- SIDEBAR (AT THE BOTTOM FOR INSTANT SYNC) ----------
    with st.sidebar:
        with st.expander("❤️ AI Heart Monitor", expanded=True):
            st.components.v1.html(
                get_ecg_heart_component(st.session_state.predicted_class, st.session_state.confidence), 
                height=450  
            )

        st.markdown("### 🧪 Download Test Samples")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            try:
                with open("normal_sample.jpg", "rb") as f:
                    st.download_button("✅ Normal", f, file_name="normal_sample.jpg", mime="image/jpeg", use_container_width=True)
            except:
                pass
        with col_s2:
            try:
                with open("abnormal_sample.jpg", "rb") as f:
                    st.download_button("⚠️ Abnormal", f, file_name="abnormal_sample.jpg", mime="image/jpeg", use_container_width=True)
            except:
                pass

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("**ECG AI Diagnosis** uses a sophisticated machine learning model to classify ECG images.")
        st.markdown('<div style="background: rgba(255, 152, 0, 0.15); border-left: 4px solid #FF9800; padding: 12px; border-radius: 12px; font-size: 0.9em; color: white;">This tool is for educational purposes only.<br><br><strong style="color:#FF9800;">Do not use for clinical decision-making.</strong></div>', unsafe_allow_html=True)