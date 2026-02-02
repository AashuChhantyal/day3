import streamlit as st
import easyocr
import fitz
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import re
import io
import base64
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Define Nepali Stop Words
NEPALI_STOPWORDS = set([
    "र", "छ", "हो", "छन्", "गरेर", "भने", "गर्न", "हुने", "पनि", "गरेको", 
    "लागि", "भएको", "गरेका", "यस", "त्यो", "सबै", "यस्तो", "गर्ने"
])

def generate_wordcloud(text):
    try:
        # 2. Setup WordCloud
        # Note: 'font_path' is crucial for Nepali. 
        # You may need to download a Nepali .ttf file and put it in your folder!
        wc = WordCloud(
            width = 800, 
            height = 400, 
            background_color = 'white',
            stopwords = NEPALI_STOPWORDS,
            font_path = 'data/NotoSans.ttf',
            colormap='viridis'
        ).generate(text)
        
        # 3. Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        return fig
    except Exception as e:
        st.warning("Could not generate Word Cloud. Ensure a Nepali .ttf font is in the /data folder.")
        return None

# Initialize models once at top to save memory
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("Sakonii/distilbert-base-nepali")
    model = AutoModel.from_pretrained("Sakonii/distilbert-base-nepali")
    # Initialize the EasyOCR reader for Nepali and English
    reader = easyocr.Reader(['ne', 'en'])
    return tokenizer, model, reader

tokenizer, nepali_model, reader = load_models()

def extracted_text_from_scanned_pdf(pdf_path):
    try:
        text = ''
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through each page of the PDF
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Render the page as an image (Pillow Image)
            pix = page.get_pixmap()
            
            # img_bytes = pix.samples

            # Create a numpy array from the image bytes
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

            # Extract text from the image using EasyOCR
            page_text = reader.readtext(img_array)

            # Concatenate the text from this page to the overall text
            text += ' '.join([item[1] for item in page_text]) + '\n'

        # Ensure the test is a string and remove leading/trailing whitespaces
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"

def read_text_from_pillow_image(image, language):
    # Initialize the EasyOCR reader for the specified language
    reader = easyocr.Reader([language])

    try:
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Read text from the image
        result = reader.readtext(img_bytes)

        # Extract and return the text
        text = '\n'.join([item[1] for item in result])

        # Ensure the text is a string and remove leading/trailing whitespaces
        return text.strip()
    except Exception as e:
        return str(e)
    
def summarize_text(model, tokenizer, text):
    # if not isinstance(text, str):
    #     test = str(text)
    if not text.strip():
        return "Text is too short to provide a meaningful summary."
    
    sentences = re.split(r'[।?\.!]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    # Too short to summarize
    if len(sentences) <= 3:
        return text

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding = True)
        with torch.no_grad():
            output = model(**inputs)

        # Get the CLS token representation
        embedding = output.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(embedding.tolist())
    # Convert to numpy array for KMeans
    embeddings_np = np.array(embeddings)

    # Clustering to find the most representative
    num_clusters = min(3, len(sentences))
    kmeans = KMeans(n_clusters = num_clusters, n_init = 10, random_state=42)
    kmeans.fit(embeddings_np)

    summary_sentences = []
    for i in range(kmeans.n_clusters):
        # Find index of the sentence closest to the cluster center 
        idx =np.where(kmeans.labels_ == i)[0][0]
        summary_sentences.append(sentences[idx])
    
    return " । ".join(summary_sentences) + " ।"

@st.cache_data
# Function to display the PDF of a given file

def displayPDF(file):
    # Opening file from file path
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf; base64, {base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit Code
st.set_page_config(page_title="Nepali Summarizer", layout="wide")

def main():
    st.title("🇳🇵 Document Summarization App")
    
    # Create data directory
    if not os.path.exists("data"):
        os.makedirs("data")

    option = st.selectbox("Choose Option", ('Pdf', 'Text', 'Image'))

    if option == 'Pdf':
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

        if uploaded_file and st.button("Summarize PDF"):
            with st.spinner("Processing PDF..."):
                filepath = os.path.join("data", uploaded_file.name)
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                col1, col2 = st.columns(2)
                with col1:
                    st.info("Original Document")
                    displayPDF(filepath)

                with col2:
                    current_text = extracted_text_from_scanned_pdf(filepath)
                    # Display the extracted text
                    st.subheader("Extracted text:")
                    st.write(current_text[:300] + "...")
                    
                    summary = summarize_text(nepali_model, tokenizer, current_text)
                    st.info("🇳🇵 AI Summary Complete")
                    st.success(summary)

                    # Add the copy-friendly block
                    st.code(summary, language="text")

                    # Store in session state so Word Cloud can access it after rerun
                    st.session_state['processed_text'] = current_text
    
    elif option == 'Text':
        user_text = st.text_area("Paste Nepali text here:", height=250)

        if st.button("Summarize Text") and user_text:
            summary = summarize_text(nepali_model, tokenizer, user_text)
            st.success(summary)
            st.code(summary, language="text")
            st.session_state['processed_text'] = user_text
    
    elif option == 'Image':
        uploaded_file = st.file_uploader("Upload your image file", type=['jpg', 'png'])

        if uploaded_file and st.button("Summarize Image"):
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                # filepath = "data/" + uploaded_file.name
                # with open(filepath, 'wb') as temp_file:
                #     temp_file.write(uploaded_file.read())
                
                # with col1:
                #     st.info("Uploaded File")
                #     image = Image.open(uploaded_file)
                #     st.image(image, caption='Uploaded Image.', use_column_width = True)
                
                # with col2:
                #     text = read_text_from_pillow_image(image, language)
                #     # Display the extracted text
                #     st.info("Extracted Text:")
                #     st.success(text)
                # Using the GLOBAL reader here for speed!
                current_text = read_text_from_pillow_image(image, 'ne')
                
                summary = summarize_text(nepali_model, tokenizer, current_text)
                st.info("Summarization Complete")
                st.success(summary)

                    # The "Copy to Clipboard"
                    # st.caption("Copy the summary below:")
                    # st.code automatically provides a 'copy' icon in the top right corner!
                st.code(summary, language="text")
                st.session_state['processed_text'] = current_text

    # --- Word Cloud Section

    if 'processed_text' in st.session_state and st.session_state['processed_text']:
        st.divider()
        if st.button("Generate Word Cloud"):
            with st.spinner("Visualizing word frequencies..."):
                fig = generate_wordcloud(st.session_state['processed_text']) # Use the 'text' variable from your OCR
                if fig:
                    st.pyplot(fig)

if __name__ == "__main__":
    main()