import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import urllib.parse

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SpaceBio Engine",
    layout="wide",
    page_icon="👨‍🚀"
)

# -----------------------------------------------------
#     UPDATED MOCK DATA FOR DATA EXPLORER
# -----------------------------------------------------
EXPLORER_DATA = [
    {
        "title": "Microgravity Gene Expression",
        "tag": "RNA-SEQ",
        "description": "Analysis of gene expression changes in Arabidopsis thaliana under microgravity conditions.",
        "year": 2023,
        "source": "NASA Ames",
        "size": "2.3GB",
        "color": "#9333ea" # Purple
    },
    {
        "title": "Cell Division in Space",
        "tag": "IMAGING",
        "description": "Time-lapse imaging of cell division processes in human cells during spaceflight.",
        "year": 2023,
        "source": "ISS",
        "size": "1.8GB",
        "color": "#ff4b4b" # Red
    },
    {
        "title": "Protein Synthesis Rates",
        "tag": "PROTEOMICS",
        "description": "Quantitative proteomics analysis of protein synthesis rates in microgravity.",
        "year": 2022,
        "source": "JAXA",
        "size": "3.1GB",
        "color": "#00C6FF" # Blue
    },
    # --- NEW DATASETS ADDED ---
    {
        "title": "Astronaut Bone Density Loss",
        "tag": "HEALTH",
        "description": "Longitudinal measurement data of bone mineral density in astronauts returning from long-duration missions.",
        "year": 2021,
        "source": "NASA HRP",
        "size": "850MB",
        "color": "#FFA500" # Orange (New color)
    },
    {
        "title": "Space-Grown Wheat Metabolome",
        "tag": "METABOLOMICS",
        "description": "Comprehensive metabolic profiling of wheat plants grown in the VEGGIE experiment facility on the ISS.",
        "year": 2024,
        "source": "CSA",
        "size": "1.1GB",
        "color": "#3CB371" # Medium Sea Green (New color)
    },
    {
        "title": "Radiation Shielding Material Tests",
        "tag": "DOSIMETRY",
        "description": "Data on radiation exposure levels measured behind various shielding materials on the lunar gateway simulator.",
        "year": 2024,
        "source": "ESA",
        "size": "5.6GB",
        "color": "#1E90FF" # Dodger Blue (New color)
    },
]

# --- 2. Data Loading and Preprocessing ---
@st.cache_data
def load_metadata(file_path):
    """Loads and preprocesses the metadata CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')
    except FileNotFoundError:
        return pd.DataFrame()

    df["Summary"] = df["Summary"].fillna("No summary provided.")
    # Use .get with a default value for safety
    df["Assay Name"] = df.get("Assay Name", pd.Series(["Unknown Assay"] * len(df)))

    if "Characteristics[Organism]" in df.columns:
        df["Organism"] = df["Characteristics[Organism]"]
    else:
        if "Organism" not in df.columns:
             df["Organism"] = pd.Series(["Unknown Organism"] * len(df))

    df["Organism"] = df["Organism"].fillna("Unknown Organism")

    if "Characteristics[Organism]" in df.columns and "Characteristics[Organism]" != "Organism":
        df = df.drop(columns=["Characteristics[Organism]"])

    return df

# --- 3. Core Logic Functions ---

def highlight_text(text: str, query: str) -> str:
    """Highlights query terms in the summary text."""
    if not query:
        return text

    query_words = [re.escape(word) for word in re.split(r'\s+', query.strip()) if word]

    highlighted_text = text
    if query_words:
        pattern = re.compile(r'\b(' + '|'.join(query_words) + r')\b', re.IGNORECASE)
        def replacer(match):
            return f"**{match.group(0)}**"

        highlighted_text = pattern.sub(replacer, text)

    return highlighted_text

def search_experiments(df, query, top_n=10, assay_filter=None, organism_filter=None, min_score=0.0):
    """Performs TF-IDF vectorization and Cosine Similarity search, applying filters and min_score."""

    filtered_df = df.copy()
    if assay_filter and assay_filter != "All":
        filtered_df = filtered_df[filtered_df["Assay Name"] == assay_filter]
    if organism_filter and organism_filter != "All":
        filtered_df = filtered_df[filtered_df["Organism"] == organism_filter]

    if filtered_df.empty or not query:
        return pd.DataFrame()

    docs = filtered_df["Summary"].tolist()

    if not any(docs) or len(docs) < 1:
          return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        doc_vectors = vectorizer.fit_transform(docs + [query])
        similarity = cosine_similarity(doc_vectors[-1], doc_vectors[:-1]).flatten()
    except ValueError:
        return pd.DataFrame()

    filtered_df = filtered_df.copy()
    filtered_df["Score"] = similarity

    filtered_df = filtered_df[filtered_df["Score"] >= min_score]

    results = filtered_df.sort_values("Score", ascending=False).head(top_n)
    return results

# --- PAGE RENDERING FUNCTIONS ---

def render_home(df_placeholder):
    """Renders the main search interface and sidebar filters. 
    It also handles the actual data loading based on sidebar input."""
    
    # ------------------ Hero Section (Centerpiece Title and Search) ------------------
    st.markdown("<div class='hero-title'><span class='gradient-text'>Centralized Space</span><br><span class='purple-gradient'>Biology Knowledge Engine</span></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a8b4; font-size: 1.1rem; margin-bottom: 50px;'>AI-powered semantic search for space biology research</p>", unsafe_allow_html=True)

    # --- Sidebar Filters & Data Loading Logic ---
    query = ""
    assay_filter = None
    organism_filter = None
    
    # Define current_df here for initial scope
    current_df = df_placeholder 

    with st.sidebar:
        st.header("🔬 Filters and Configuration")
        default_path = "data/OSD-101_clean.csv"    # ✅ Relative path
        file_path = st.text_input(
            "Enter CSV file path:",
            default_path,
            help="The CSV must contain 'Summary', 'Assay Name', and 'Organism' columns.",
            key="sidebar_filepath"
        )
        
        # Load the DF based on the path provided in the sidebar
        current_df = load_metadata(file_path)

        # CORRECTED INDENTATION FOR THE IF/ELSE BLOCK
        if current_df.empty:
            st.warning("Please check the file path. Could not load data.")
            # query remains ""
        else:
            st.markdown("---")
            st.subheader("Refine Search Results")
            assay_types = ["All"] + sorted(current_df["Assay Name"].dropna().unique().tolist())
            assay_filter = st.selectbox("Filter by Assay Type:", assay_types)
            if assay_filter == "All": assay_filter = None

            organisms = ["All"] + sorted(current_df["Organism"].dropna().unique().tolist())
            organism_filter = st.selectbox("Filter by Organism:", organisms)
            if organism_filter == "All": organism_filter = None

            st.markdown("---")
            st.info(f"Loaded records: **{len(current_df)}**")

    # The actual DataFrame to use in the main app is the one loaded in the sidebar
    df = current_df
    
    # --- Main Search Area (Centralized) ---
    if not df.empty:
        col_left, col_center, col_right = st.columns([1, 3, 1])
        with col_center:
            query = st.text_input(
                "🔍 Search experiments, datasets, publications...",
                placeholder="Search experiments, datasets, publications...",
                label_visibility="collapsed"
            )

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

    min_score = 0.0

    # --- Results Section ---
    if query and not df.empty:
        with st.spinner("Searching and ranking experiments..."):
            results = search_experiments(
                df,
                query,
                top_n=10,
                assay_filter=assay_filter,
                organism_filter=organism_filter,
                min_score=min_score
            )

        if not results.empty:
            st.markdown(f"### Top {len(results)} Relevant Results")
            for idx, row in results.iterrows():
                score_html = f'<span class="score-badge">Similarity: {row["Score"]:.4f}</span>'
                title_html = f"**{row.get('Assay Name', 'No Title')}** {score_html}"

                with st.expander(title_html, expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Organism:** `{row.get('Organism', 'Unknown')}`")
                    with col2:
                        st.markdown(f"**Raw Data Files:** `{row.get('Raw Data File', 'N/A')}`")
                    st.markdown("---")
                    st.markdown("#### Experiment Summary")
                    summary_text = row.get('Summary', 'No summary available.')
                    st.markdown(highlight_text(summary_text, query))
                    st.markdown(f"**Source Description:** {row.get('Comment[Source Description]', 'N/A')}")


            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Top Results as CSV",
                data=csv,
                file_name="top_results.csv",
                mime="text/csv"
            )

        else:
            st.info("No matching experiments found for your query with the current filters.")
    elif not df.empty:
        st.markdown("<h4 style='text-align: center; color: #a0a8b4;'>Start your research by entering a query above.</h4>", unsafe_allow_html=True)

def render_features():
    st.header("✨ Key Features of SpaceBio Engine")
    st.markdown("""
    <div style='background-color: #161b22; padding: 20px; border-radius: 10px;'>
        <p>This engine provides advanced semantic search, filtering by organism and assay type, and interactive result cards for deep dives into space biology data.</p>
        <ul>
            <li>**Semantic Search:** Uses Cosine Similarity on TF-IDF vectors for highly relevant results.</li>
            <li>**Data Filtering:** Quickly narrow down results using sidebar filters.</li>
            <li>**High-Contrast UI:** Professional dark theme for comfortable long-term use.</li>
            <li>**Downloadable Results:** Export search results as CSV for further analysis.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_explorer_card(item):
    """Generates the HTML/CSS for a single data explorer card."""
    tag_color = item.get("color", "#ff4b4b")

    html_card = f"""
    <div class='explorer-card-container'>
        <div class='card-title-row'>
            <div class='card-title'>{item['title']}</div>
            <span class='card-tag' style='border: 1px solid {tag_color}; color: {tag_color};'>{item['tag']}</span>
        </div>
        <p class='card-description'>{item['description']}</p>
        <div class='card-footer'>
            <span class='card-footer-item'>🗓️ {item['year']}</span>
            <span class='card-footer-item'>🧑‍🚀 {item['source']}</span>
            <span class='card-footer-item'>⬇️ {item['size']}</span>
        </div>
    </div>
    """
    return html_card

def render_data_explorer():
    """Renders the Data Explorer section with interactive-looking tabs and cards."""

    st.markdown("<h2 class='explorer-main-title'>Data Explorer</h2>", unsafe_allow_html=True)

    # 1. Simulate the explorer tabs
    col_tabs = st.columns([1, 1, 1, 3])
    with col_tabs[0]:
        st.markdown("<div class='explorer-tab active-tab'>Experiments</div>", unsafe_allow_html=True)
    with col_tabs[1]:
        st.markdown("<div class='explorer-tab inactive-tab'>Datasets</div>", unsafe_allow_html=True)
    with col_tabs[2]:
        st.markdown("<div class='explorer-tab inactive-tab'>Publications</div>", unsafe_allow_html=True)

    st.markdown("---") # Visual separator below the tabs

    # 2. Render the Cards using st.columns. Now using two rows for 6 items.
    st.markdown("### Featured Data Packages")

    # First row: 3 cards
    cols1 = st.columns(3)
    for i in range(3):
        with cols1[i]:
            st.markdown(create_explorer_card(EXPLORER_DATA[i]), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Space between rows

    # Second row: 3 cards
    cols2 = st.columns(3)
    for i in range(3, 6):
        with cols2[i-3]:
            st.markdown(create_explorer_card(EXPLORER_DATA[i]), unsafe_allow_html=True)


def render_about():
    st.header("🧑‍💻 About SpaceBio Engine")
    st.markdown("""
    <div style='background-color: #161b22; padding: 20px; border-radius: 10px;'>
        <p>The SpaceBio Engine was created as a project to demonstrate an efficient and user-friendly interface for searching complex scientific metadata, specifically focusing on space biology and 'omics' data.</p>
        <p>It leverages Python libraries like Streamlit, Pandas, and Scikit-learn for the front-end interface, data processing, and search algorithm, respectively.</p>
    </div>
    """, unsafe_allow_html=True)

def render_contact():
    """Renders the Contact Us section using card-based layout with custom details."""

    # Custom contact details
    user_email = "shubhamramdhiraj@gmail.com"
    user_phone = "+91 9315183970" # Formatting phone number for better readability

    st.markdown("<h2 class='contact-main-title'>Get in Touch</h2>", unsafe_allow_html=True)

    # --- 1. Email and Phone Cards ---
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"""
        <div class='contact-card'>
            <div class='contact-icon-row'>
                <span class='contact-icon email-icon'>📧</span>
                <span class='contact-label'>Email</span>
            </div>
            <p class='contact-detail'>{user_email}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='contact-card'>
            <div class='contact-icon-row'>
                <span class='contact-icon phone-icon'>📞</span>
                <span class='contact-label'>Phone</span>
            </div>
            <p class='contact-detail'>{user_phone}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer

    # --- 2. Address Card (Takes up 2 columns) ---
    col4, col5 = st.columns([4, 1])

    with col4:
        st.markdown("""
        <div class='contact-card address-card'>
            <div class='contact-icon-row'>
                <span class='contact-icon address-icon'>📍</span>
                <span class='contact-label'>Address</span>
            </div>
            <p class='contact-detail'>NASA Ames Research Center<br>Moffett Field, CA 94035</p>
        </div>
        """, unsafe_allow_html=True)


# --- Inject custom CSS for a dark, professional, and centralized layout
st.markdown("""
    <style>
    /* 1. Core App Styling */
    .stApp { background-color: #0d1117; color: #ffffff; }
    .stApp > header { display: none; }
    .main > div { background-color: transparent; }
    .block-container { padding-top: 2rem; }
    .stSidebar { background-color: #161b22; }

    /* 2. Streamlit Tabs Styling (Unchanged) */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; justify-content: center; border-bottom: 1px solid #2f363d; padding-bottom: 0; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #a0a8b4; border-bottom: none !important; font-weight: 600; transition: color 0.2s; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { color: #00C6FF !important; border-bottom: 3px solid #00C6FF !important; background-color: transparent; }

    /* 3. Hero Section Styling (Unchanged) */
    .hero-title { text-align: center; font-size: 3.5rem; font-weight: 900; margin-top: 50px; margin-bottom: 15px; line-height: 1.2; }
    .gradient-text { color: transparent; background: linear-gradient(90deg, #00C6FF, #0072FF); -webkit-background-clip: text; background-clip: text; }
    .purple-gradient { background: linear-gradient(90deg, #9333ea, #6b21a8); -webkit-background-clip: text; background-clip: text; color: transparent; }

    /* 4. Search Input Styling (Padded) */
    .stTextInput>div>div>input {
        background-color: #161b22;
        border: 1px solid #3b82f6;
        color: white;
        border-radius: 30px;
        padding: 10px 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        height: auto;
    }
    .stTextInput { max-width: 600px; margin: 30px auto; }

    /* 5. Heading Enhancements (Unchanged) */
    h1, h2, h3, h4, .stSidebar h2, .stSidebar h3 { font-weight: 900 !important; }
    .stSidebar .st-emotion-cache-1cypcdp,
    .stSidebar .st-emotion-cache-16niy5w,
    section.main h2,
    section.main h3 { font-weight: 900 !important; color: #00C6FF !important; padding-bottom: 0.3em; border-bottom: 2px solid #2f363d; margin-bottom: 15px; }
    .streamlit-expanderContent h4 { color: #f0f2f6; font-weight: 800 !important; border-bottom: none; }

    /* 6. Component Styling (Unchanged) */
    .streamlit-expanderHeader { background-color: #161b22; color: #f0f2f6; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4); margin: 12px 0; padding: 18px; border-left: 6px solid #3b82f6; font-weight: bold; font-size: 1.1rem; }
    .streamlit-expanderHeader:hover { background-color: #1f2730; border-left: 6px solid #ff4b4b; }
    .score-badge { background-color: #ff4b4b; color: white; padding: 6px 10px; border-radius: 16px; font-size: 0.85rem; font-weight: bold; margin-left: 15px; min-width: 100px; text-align: center; }

    /* 7. DATA EXPLORER SPECIFIC STYLING (Unchanged hover) */
    .explorer-main-title { text-align: center; font-size: 2.5rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #9333ea, #6b21a8); -webkit-background-clip: text; background-clip: text; color: transparent; margin-bottom: 40px; border-bottom: none !important; }
    .explorer-tab { text-align: center; padding: 10px 20px; margin: 0 5px; border-radius: 10px; cursor: pointer; font-weight: 600; transition: all 0.3s; }
    .active-tab { background: linear-gradient(135deg, #00C6FF, #0072FF); color: white; box-shadow: 0 4px 10px rgba(0, 114, 255, 0.4); border: none; }
    .inactive-tab { background-color: #161b22; color: #a0a8b4; border: 1px solid #2f363d; }
    .inactive-tab:hover { background-color: #1f2730; color: white; }
    .explorer-card-container { background-color: #161b22; padding: 25px; border-radius: 15px; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5); border-top: 5px solid #00C6FF; transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s; }
    .explorer-card-container:hover {
        transform: translateY(-4px);
        border-top: 5px solid #ff4b4b;
        box-shadow: 0 10px 25px rgba(255, 75, 75, 0.4);
    }
    .card-title-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .card-title { font-size: 1.2rem; font-weight: 800; color: #f0f2f6; }
    .card-tag { font-size: 0.7rem; font-weight: bold; padding: 4px 10px; border-radius: 5px; white-space: nowrap; }
    .card-description { color: #a0a8b4; font-size: 0.95rem; margin-bottom: 20px; min-height: 50px; }
    .card-footer { display: flex; justify-content: space-between; padding-top: 15px; border-top: 1px solid #2f363d; }
    .card-footer-item { font-size: 0.85rem; color: #00C6FF; font-weight: 600; }

    /* 8. CONTACT US SPECIFIC STYLING (Unchanged hover) */
    .contact-main-title {
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #9333ea, #6b21a8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 40px;
        border-bottom: none !important;
    }

    .contact-card {
        background-color: #161b22;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid #2f363d;
        transition: all 0.3s, border-color 0.3s, box-shadow 0.3s;
        height: 100%;
    }
    .contact-card:hover {
        border-color: #00C6FF;
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 198, 255, 0.3);
    }

    .address-card {
        margin-top: 0;
    }

    .contact-icon-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }

    .contact-icon {
        font-size: 1.8rem;
        margin-right: 15px;
        color: #00C6FF;
    }

    .contact-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f0f2f6;
    }

    .contact-detail {
        font-size: 1.1rem;
        color: #a0a8b4;
        margin-left: 55px;
        margin-bottom: 0;
    }

    </style>
    """, unsafe_allow_html=True)


# --- 4. Streamlit App Interface (Main App Logic) ---
def main_app():

    # Static header above the tabs
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #2f363d;">
            <h3 style="color: #00C6FF; margin: 0; font-weight: 900 !important;"><img src="https://placehold.co/24x24/00C6FF/white?text=%E2%9C%88" style="vertical-align: middle; margin-right: 5px;"> SpaceBio Engine</h3>
        </div>
        """, unsafe_allow_html=True)

    # Define tabs
    tab_titles = ["Home", "Features", "Data Explorer", "About", "Contact"]
    home_tab, features_tab, data_explorer_tab, about_tab, contact_tab = st.tabs(tab_titles)

    # We pass a dummy/default DataFrame, but render_home will handle the *actual*
    # load based on the sidebar path.
    default_df = pd.DataFrame() # An empty df for initial setup outside render_home's scope

    # --- Render content based on active tab ---

    with home_tab:
        # render_home now contains the logic for the sidebar filters, search, and results
        render_home(default_df)

    with features_tab:
        render_features()

    with data_explorer_tab:
        render_data_explorer()

    with about_tab:
        # Indentation fix applied here
        render_about() 

    with contact_tab:
        render_contact()

if __name__ == "__main__":
    main_app()
