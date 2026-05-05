import streamlit as st
from medical_summarizer_final import Summarizer, MODELS, filter_hallucinations, build_structured_summary, Report

# Configure the page
st.set_page_config(page_title="Medical Report Summarizer", page_icon="🏥", layout="wide")

st.title("🏥 Medical Report Summarizer")
st.write("Upload a medical report text file or paste the text directly to get a structured and AI-generated summary.")

# Cache the model loading so it doesn't reload on every interaction
@st.cache_resource(show_spinner="Loading model... This may take a minute.")
def load_model(model_name):
    # MODELS mapping is defined in the original script
    model_id = MODELS.get(model_name, "t5-small")
    return Summarizer(model_id)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Select Model", list(MODELS.keys()))

summarizer = load_model(model_choice)

# Input section
input_method = st.radio("Choose input method:", ("Paste Text", "Upload Text File"))

report_text = ""
filename = "Pasted_Text.txt"

if input_method == "Upload Text File":
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        report_text = uploaded_file.getvalue().decode("utf-8")
        filename = uploaded_file.name
else:
    report_text = st.text_area("Paste medical report text here:", height=300)

# Summarize button
if st.button("Summarize Report", type="primary"):
    if report_text.strip():
        with st.spinner(f"Summarizing using {model_choice}..."):
            try:
                # Create Report object
                report = Report(filename=filename, text=report_text)
                
                # Generate summary using the imported class
                raw_summary = summarizer.summarize(report.text)
                final_summary = filter_hallucinations(raw_summary, report.text)
                
                # Build structured output
                structured_output = build_structured_summary(report, final_summary)
                
                st.success("Summary Generated Successfully!")
                
                # Display the output
                st.subheader("Results")
                st.text_area("Structured Summary Output:", value=structured_output, height=400)
                
                # Option to download the summary
                st.download_button(
                    label="Download Summary",
                    data=structured_output,
                    file_name=f"{filename.split('.')[0]}_summary.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
    else:
        st.warning("Please provide a medical report to summarize first.")
