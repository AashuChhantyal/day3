## Nepali Text & Document Summarizer 🇳🇵
An AI-powered application designed to summarize Nepali content from various sources, including plain text, images (OCR), and PDF documents.
Built with Streamlit and state-of-the-art Transformers.

## 🚀 Features:
Text Summarization: Paste Nepali text directly to get a concise summary.

OCR Support: Upload images (PNG, JPG) of Nepali text and extract content using EasyOCR.

PDF Integration: Upload PDF files to summarize long documents using PyMuPDF.

Visualizations: Generates word clouds and frequency charts for the processed text.

Deep Learning: Utilizes fine-tuned Transformer models for accurate Nepali context.
## 🛠️ Installation
1. Clone the repository:
```
Bash
git clone https://github.com/AashuChhantyal/day3.git
cd day3
```
2. Set up a Virtual Environment:
```
Bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
```

3. Install Dependencies:
```
Bash
pip install streamlit easyocr pymupdf transformers torch scikit-learn pillow numpy matplotlib wordcloud
```
## 💻 Usage
To launch the application, run the following command from the root directory:
```
Bash
streamlit run Nepali_Summarizer/aep_summarize.py
```
