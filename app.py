import streamlit as st
import streamlit.components.v1 as components
import time
import os
from PIL import Image, ImageOps
from ultralytics import YOLO
from google import genai

# ==========================================
# 1. PAGE ARCHITECTURE & THEME
# ==========================================
st.set_page_config(
    page_title="Citrus Pathology System", 
    page_icon="🍊",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 🔥 INVISIBLE TOP ANCHOR FOR THE 'BACK TO TOP' BUTTON
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

# Minimalist CSS Injection
st.markdown("""
    <style>
    /* Global Typography & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; font-size: 15px; background-color: #f8fafc; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1000px; }
    
    /* Hide Streamlit Footer */
    footer {visibility: hidden;}

    /* ---> FAVY JAY NATIVE HEADER INJECTION <--- */
    .favy-header {
        position: fixed;
        top: 0.8rem;
        left: 4.5rem;
        z-index: 999999; 
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: 0.5px;
        pointer-events: none; 
    }
    .favy-header span {
        font-weight: 400;
        color: #64748b;
        font-size: 0.95rem;
    }

    /* Hero Section */
    .hero-title { font-size: 2.8rem; font-weight: 800; color: #0f172a; text-align: center; margin-bottom: 0.2rem; margin-top: 1.5rem; letter-spacing: -0.02em; }
    .hero-subtitle { font-size: 1.1rem; text-align: center; color: #64748b; margin-bottom: 2.5rem; font-weight: 400; }
    
    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 3.5rem; white-space: pre-wrap; font-weight: 600; color: #64748b; }
    .stTabs [aria-selected="true"] { color: #0ea5e9 !important; border-bottom: 3px solid #0ea5e9 !important; }

    /* Uploader Dropzone Styling */
    div[data-testid="stFileUploader"] section {
        padding: 3rem 2rem !important;
        text-align: center !important;
        background-color: #ffffff !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] section:hover { border-color: #0ea5e9 !important; background-color: #f0f9ff !important; }
    
    div[data-testid="stFileUploader"] section button {
        background-color: #0ea5e9 !important; color: white !important;
        border-radius: 50px !important; font-weight: 600 !important;
        padding: 0.5rem 2rem !important; margin-top: 1rem;
    }

    /* ========================================================
       🔥 INITIAL BUTTON & IMAGE LAYOUT (Restored)
       ======================================================== */
    div[data-testid="stTabs"] div.stButton > button {
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 0.2rem 0.5rem !important;
        background-color: #ffffff !important;
        color: #555555 !important;
        border: 1px solid #e0e0e0 !important;
        width: 100% !important;
    }
    div[data-testid="stTabs"] div.stButton > button:hover {
        border-color: #0066fe !important;
        color: #0066fe !important;
    }

    /* Premium Reset Button Base Settings (Kept independent) */
    .reset-btn-container div.stButton > button {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        color: #0284c7 !important;
        border: 1px solid #bae6fd !important;
        border-radius: 12px !important; 
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        padding: 0.7rem 0 !important; 
        width: 100% !important;
        height: auto !important;
        box-shadow: 0 2px 4px rgba(14, 165, 233, 0.1) !important;
        transition: all 0.2s ease !important;
        margin-top: 0.5rem !important;
    }
    .reset-btn-container div.stButton > button:hover { 
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: #ffffff !important; 
        border-color: #0ea5e9 !important; 
        transform: translateY(-2px) !important; 
        box-shadow: 0 6px 12px rgba(14, 165, 233, 0.25) !important; 
    }

    /* 📱 MOBILE OPTIMIZATIONS (Restored exactly to your initial code) */
    @media (max-width: 576px) {
        .favy-header { left: 3.5rem; font-size: 1rem; top: 0.9rem; }
        .hero-title { font-size: 2rem; margin-top: 1rem; }
        .hero-subtitle { font-size: 0.95rem; margin-bottom: 1.5rem; }
        
        div[data-testid="column"] {
            width: 25% !important;
            flex: 1 1 25% !important;
            min-width: 25% !important;
            padding: 0 0.1rem !important;
        }
        div.stButton > button {
            font-size: 0.5rem !important;
            padding: 0.1rem 0 !important;
            min-height: 1.5rem !important;
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE & MODEL INITIALIZATION
# ==========================================
if "active_image" not in st.session_state: st.session_state.active_image = None
if "sample_label" not in st.session_state: st.session_state.sample_label = None
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 

client = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Background API uninitialized: {e}")

@st.cache_resource
def load_yolo_model(): return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("Critical Error: Could not load model weights (best.pt).")
    st.stop()

# ==========================================
# 3. HEADER, SIDEBAR & INPUT ROUTING
# ==========================================

# Inject the Brand Text
st.markdown('<div class="favy-header">Favy Jay <span>| System Architecture</span></div>', unsafe_allow_html=True)

# Hero Section
st.markdown('<div class="hero-title">Citrus Pathology AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload a fruit image for instant computer vision diagnostics and agronomist treatment plans.</div>', unsafe_allow_html=True)

# Config Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    conf_threshold = st.slider("Confidence Score Threshold", 0.05, 1.00, 0.25, 0.05)
    
    st.markdown("---")
    st.info("""
    **ℹ️ About the System**
    
    This AI engine detects critical citrus diseases (Canker, Greening, Melanose) using an optimized **YOLO26** architectural framework.
    
    *Adjust the slider above to fine-tune the model's detection sensitivity.*
    """)

# The Tabbed Interface
tab1, tab2 = st.tabs(["📤 Upload Custom Scan", "🖼️ Try Demo Samples"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}")
    
    # Premium Reset Button Container
    st.markdown('<div class="reset-btn-container">', unsafe_allow_html=True)
    if st.button("🔄 Clear System & Reset Dashboard", use_container_width=True):
        st.session_state.active_image = None
        st.session_state.sample_label = None
        st.session_state.uploader_key += 1 
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Restored to exact initial 4-column layout
    col1, col2, col3, col4 = st.columns(4)
    
    def render_sample_card(column, file_name, display_name):
        with column:
            file_exists = os.path.exists(file_name)
            if file_exists:
                raw_img = Image.open(file_name)
            else:
                raw_img = Image.new('RGB', (190, 190), color=(226, 232, 240))
                
            square_thumb = ImageOps.fit(raw_img, (190, 190), Image.Resampling.LANCZOS)
            
            st.image(square_thumb, use_container_width=True)
            
            # Button restored to its exact initial behavior (below image, visible)
            if st.button(display_name, key=f"btn_{display_name.lower()}", disabled=not file_exists):
                st.session_state.active_image = file_name
                st.session_state.sample_label = f"{display_name} Sample Profile"
            
            if not file_exists:
                st.caption(f"⚠️ Missing file")

    render_sample_card(col1, "images/sample_canker.jpg", "Canker")
    render_sample_card(col2, "images/sample_fresh.jpg", "Fresh")
    render_sample_card(col3, "images/sample_greening.jpg", "Greening")
    render_sample_card(col4, "images/sample_melanose.jpg", "Melanose")

# Resolve input image
final_input_image = None
if uploaded_file is not None:
    final_input_image = Image.open(uploaded_file)
    st.session_state.active_image = None
    st.session_state.sample_label = None
elif st.session_state.active_image is not None:
    final_input_image = Image.open(st.session_state.active_image)

# ==========================================
# 4. EXPERT COGNITIVE ENGINE
# ==========================================
def get_agricultural_remedy(disease_name):
    if not client: return "⚠️ API Key missing."
    prompt = f"""
    You are an expert plant pathologist. The YOLO26 model detected '{disease_name}' on a citrus fruit.
    Provide a professional, structured remedy plan. Use ONLY these exact markdown headers:
    ### 📋 Immediate Field Actions
    ### 🧪 Treatment Options (Organic & Chemical)
    ### 🛡️ Long-Term Prevention
    Keep it actionable, scientific, and concise. No conversational fluff.
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e: return f"❌ API Error: {e}"

# ==========================================
# 5. INFERENCE & RESULTS DASHBOARD
# ==========================================
if final_input_image is not None:
    # Set the precise anchor right above the results
    render_key = str(int(time.time() * 1000))
    st.markdown(f'<div id="results-target-{render_key}"></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.spinner("Processing computer vision tensors..."):
        results = model.predict(final_input_image, conf=conf_threshold)
        annotated_image = results[0].plot(pil=True)
        detected_boxes = results[0].boxes
    
    res_col1, res_col2 = st.columns([1.2, 1.5], gap="large")
    
    # LEFT COLUMN: Visuals
    with res_col1:
        st.markdown("#### 🔬 Neural Vision Scan")
        st.image(annotated_image, use_container_width=True, caption="YOLO26 Feature Extraction Matrix")
        if st.session_state.sample_label:
            st.caption(f"Active Session: **{st.session_state.sample_label}**")

    # RIGHT COLUMN: Diagnostics
    with res_col2:
        st.markdown("#### 📊 Diagnostic Report")
        
        if len(detected_boxes) == 0:
            st.info("🟢 **STATUS CLEAR:** No pathological signatures detected at current confidence threshold.")
        else:
            found_labels = [model.names[int(box.cls[0])] for box in detected_boxes]
            unique_labels = list(set(found_labels))
            
            for condition in unique_labels:
                if condition.lower() == 'fresh':
                    st.success("✅ **DIAGNOSIS:** FRESH / HEALTHY FRUIT\n\nSurface parameters match baseline. No routing required.")
                else:
                    if condition.lower() == 'canker': st.error("🚨 **DIAGNOSIS:** CITRUS CANKER DETECTED")
                    elif condition.lower() == 'melanose': st.warning("⚠️ **DIAGNOSIS:** MELANOSE DETECTED")
                    elif condition.lower() == 'greening': st.error("☢️ **DIAGNOSIS:** CITRUS GREENING (HLB) DETECTED")
                    
                    with st.spinner(f"Consulting agronomy database for {condition}..."):
                        remedy_text = get_agricultural_remedy(condition)
                    
                    with st.expander(f"📖 View Expert Treatment Plan for {condition.capitalize()}", expanded=True):
                        st.markdown(remedy_text)
    
    # 🚀 DELAYED AUTO-SCROLL
    components.html(f"""
        <script>
            setTimeout(() => {{
                const target = window.parent.document.getElementById('results-target-{render_key}');
                if(target) {{
                    target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                }}
            }}, 800);
        </script>
    """, height=0)

# ==========================================
# 6. FLOATING 'BACK TO TOP' BUTTON INJECTION
# ==========================================
components.html("""
    <script>
        // 1. Create the button if it doesn't exist
        let btn = window.parent.document.getElementById('favy-btt-btn');
        if (!btn) {
            btn = window.parent.document.createElement('button');
            btn.id = 'favy-btt-btn';
            btn.innerHTML = '↑ Top';
            
            btn.style.cssText = 'position:fixed; bottom:30px; right:30px; z-index:999999; background:#0f172a; color:white; border:none; padding:12px 20px; border-radius:50px; cursor:pointer; box-shadow:0 8px 16px rgba(0,0,0,0.2); font-weight:700; font-family:sans-serif; font-size: 14px; transition: all 0.3s ease;';
            
            btn.onmouseover = () => { btn.style.background = '#0ea5e9'; btn.style.transform = 'translateY(-3px)'; };
            btn.onmouseout = () => { btn.style.background = '#0f172a'; btn.style.transform = 'translateY(0)'; };
            
            window.parent.document.body.appendChild(btn);
        }
        
        // 2. ALWAYS RE-BIND THE CLICK EVENT on every model run to prevent stale references
        btn.onclick = () => {
            // Target 1: Streamlit's specific internal view container
            const viewContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (viewContainer) { viewContainer.scrollTo({top: 0, behavior: 'smooth'}); }
            
            // Target 2: The classic main container
            const mainContainer = window.parent.document.querySelector('.main');
            if (mainContainer) { mainContainer.scrollTo({top: 0, behavior: 'smooth'}); }
            
            // Target 3: Global Window Fallback
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
            
            // Target 4: Magnetically pull to the top anchor
            const topAnchor = window.parent.document.getElementById('top-anchor');
            if (topAnchor) { topAnchor.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        };
    </script>
""", height=0)