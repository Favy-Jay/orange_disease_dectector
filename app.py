import streamlit as st
import streamlit.components.v1 as components  # <-- Added for JS auto-scroll
from PIL import Image, ImageOps
from ultralytics import YOLO
from google import genai

# 1. Page Layout Architecture
st.set_page_config(
    page_title="Citrus Pathology System", 
    page_icon="🍊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Minimalist CSS Injection
st.markdown("""
    <style>
    /* Slightly reduce overall typography scale */
    html, body, .stApp {
        font-size: 15px;
    }

    /* Keep content clear of fixed header */
    .main .block-container {
        padding-top: 4.25rem;
    }

    /* Centered Title Styling */
    .main-title { font-size: 2.35rem; font-weight: 800; color: #222222; text-align: center; margin-top: 1rem; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.03rem; text-align: center; color: #666666; margin-bottom: 2.1rem; }
    .sample-title { font-size: 0.98rem; font-weight: 600; text-align: center; color: #444444; margin-top: 1.6rem; margin-bottom: 0.9rem; }
    
    /* Replicating the bright blue upload button aesthetic */
    div[data-testid="stFileUploader"] section button {
        background-color: #0066fe !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2.5rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        box-shadow: 0 4px 12px rgba(0, 102, 254, 0.2);
    }
    
    /* Center aligning the dropzone text element */
    div[data-testid="stFileUploader"] section {
        padding: 2rem 1rem !important;
        text-align: center !important;
        background-color: #fafafa !important;
        border: 2px dashed #e0e0e0 !important;
        border-radius: 16px !important;
        max-width: 550px;
        margin: 0 auto !important;
    }
    
    /* Styling for the clean tryout selection buttons below */
    div.stButton > button {
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 0.2rem 0.5rem !important;
        background-color: #ffffff !important;
        color: #555555 !important;
        border: 1px solid #e0e0e0 !important;
    }
    div.stButton > button:hover {
        border-color: #0066fe !important;
        color: #0066fe !important;
    }

    /* Plain label near the top-right sidebar area */
    .app-label {
        position: fixed;
        top: 0.7rem;
        right: 3.2rem;
        z-index: 1001;
        color: #1f2937;
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1.2;
        pointer-events: none;
    }

    /* ---> MOBILE OPTIMIZATION: Force the 4 sample columns to stay in a single row <--- */
    @media (max-width: 576px) {
        div[data-testid="column"] {
            width: 25% !important;
            flex: 1 1 25% !important;
            min-width: 25% !important;
            padding: 0 0.2rem !important;
        }
        div.stButton > button {
            font-size: 0.6rem !important;
            padding: 0.2rem 0.1rem !important;
            min-height: 2rem !important;
        }
        .sample-title {
            margin-top: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-label">Orange Disease Detector</div>', unsafe_allow_html=True)

# Hero Section
st.markdown('<div class="main-title">Upload an image to detect orange diseases</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">or drop a file to automatically generate expert agronomy treatment summaries.</div>', unsafe_allow_html=True)

# 2. State Engine Initialization
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "sample_label" not in st.session_state:
    st.session_state.sample_label = None

# 3. Model Dependencies & API Handshake
client = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Failed to authenticate background Gemini client: {e}")

@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Could not load model weights. Error: {e}")

# 4. TOP ELEMENT: File Input Component
uploaded_file = st.file_uploader(
    label="Upload Canvas Input Dropzone", 
    type=["jpg", "jpeg", "png"], 
    label_visibility="collapsed"
)

# 5. BOTTOM ELEMENT: Equal-Width Square Sample Section
st.markdown('<div class="sample-title">No image? Try one of these:</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

def render_sample_card(column, file_name, display_name):
    """Helper function to load an image, crop it to a uniform 1:1 square, and display its trigger."""
    with column:
        try:
            raw_img = Image.open(file_name)
            # Hard-forces any aspect ratio into a clean square thumbnail
            square_thumb = ImageOps.fit(raw_img, (190, 190), Image.Resampling.LANCZOS)
            st.image(square_thumb, use_container_width=True)
            if st.button(display_name, key=f"btn_{display_name.lower()}"):
                st.session_state.active_image = file_name
                st.session_state.sample_label = f"{display_name} Sample Profile"
        except:
            st.caption(f"Missing {file_name}")

render_sample_card(col1, "images/sample_canker.jpg", "Canker")
render_sample_card(col2, "images/sample_fresh.jpg", "Fresh")
render_sample_card(col3, "images/sample_greening.jpg", "Greening")
render_sample_card(col4, "images/sample_melanose.jpg", "Melanose")

# 6. Primary Input Routing Queue
final_input_image = None

# Custom user upload completely overrides selected samples
if uploaded_file is not None:
    final_input_image = Image.open(uploaded_file)
    st.session_state.active_image = None
    st.session_state.sample_label = None
elif st.session_state.active_image is not None:
    final_input_image = Image.open(st.session_state.active_image)

# Sidebar Configuration Utilities
st.sidebar.header("🔧 Inference Controls")
conf_threshold = st.sidebar.slider("Model Confidence Threshold", 0.05, 1.00, 0.25, 0.05)

if st.session_state.active_image or uploaded_file:
    if st.sidebar.button("🔄 Clear System Status"):
        st.session_state.active_image = None
        st.session_state.sample_label = None
        st.rerun()

# 7. Cognitive Engine Prompt Architecture
def get_agricultural_remedy(disease_name):
    if not client:
        return "⚠️ Background API Client uninitialized. Ensure key is placed in .streamlit/secrets.toml."
    prompt = f"""
    You are an expert agricultural scientist and plant pathologist specializing in citrus orchard management.
    The computer vision model has detected '{disease_name}' on an orange fruit.
    Provide a concise, professional, and actionable remedy plan for a farmer. Structure your response EXACTLY with these markdown sections:
    ### 📋 Immediate Field Actions
    ### 🧪 Treatment & Control Options
    - **Organic/Cultural:** - **Chemical (if severe):** ### 🛡️ Preventative Measures
    Keep explanations scientific yet highly practical. Do not include conversational intro or outro text.
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"❌ Failed to generate remedy recommendation. Error: {e}"

# 8. Evaluation & Execution Engine Output
if final_input_image is not None:
    st.write("---")
    
    # ---> SCROLL TARGET 1: Auto-Scroll to Detection Section <---
    st.markdown('<div id="detection-target"></div>', unsafe_allow_html=True)
    components.html("""
        <script>
            var target = window.parent.document.getElementById('detection-target');
            if (target) { target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
    """, height=0)

    if st.session_state.sample_label:
        st.caption(f"🚀 Active Session: Running **{st.session_state.sample_label}**")
        
    with st.spinner("Executing spatial anomaly detection..."):
        results = model.predict(final_input_image, conf=conf_threshold)
        
    annotated_image = results[0].plot(pil=True)
    st.write("### 🔍 Vision Model Performance Layer:")
    st.image(annotated_image, use_container_width=True)
    
    st.write("---")
    
    # ---> SCROLL TARGET 2: Auto-Scroll to Remedy Section <---
    st.markdown('<div id="remedy-target"></div>', unsafe_allow_html=True)
    st.write("### 📋 Diagnostic & Remedy Matrix Summary")
    
    # Execute the scroll immediately before querying Gemini so the user watches the remedies generate
    components.html("""
        <script>
            var target = window.parent.document.getElementById('remedy-target');
            if (target) { target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
    """, height=0)
    
    detected_boxes = results[0].boxes
    
    if len(detected_boxes) == 0:
        st.info("ℹ️ **STATUS:** The model could not identify any disease or healthy fruit signatures at this confidence level.")
    else:
        found_labels = [model.names[int(box.cls[0])] for box in detected_boxes]
        unique_labels = list(set(found_labels))
        
        for condition in unique_labels:
            if condition.lower() == 'fresh':
                st.success("🟢 **DIAGNOSIS: FRESH (HEALTHY FRUIT PROFILE)**")
                st.write("Surface parameters match a healthy index baseline. No active chemical routing required.")
            else:
                if condition.lower() == 'canker': st.error("🔴 **DIAGNOSIS DETECTED: CITRUS CANKER**")
                elif condition.lower() == 'melanose': st.warning("🟡 **DIAGNOSIS DETECTED: MELANOSE**")
                elif condition.lower() == 'greening': st.error("🟠 **DIAGNOSIS DETECTED: CITRUS GREENING (HLB)**")
                
                if client:
                    with st.spinner(f"Routing '{condition}' signatures to background treatment engine..."):
                        remedy_text = get_agricultural_remedy(condition)
                    st.write(remedy_text)
                    st.write("---")