🏥 Medical Report Summarization System

A transformer-based NLP project that automatically generates concise and structured summaries from medical reports to assist healthcare professionals in quick decision-making.

📌 Overview

Medical reports are often lengthy and complex. This project uses Transformer-based models (T5) to generate abstractive summaries and extract key clinical information such as:

Symptoms
Diagnosis
Tests / Investigations
Treatment

The system also evaluates summary quality using ROUGE metrics and provides a user interface for easy interaction.

🚀 Features
🔹 Abstractive medical report summarization
🔹 Structured output (Symptoms, Diagnosis, Tests, Treatment)
🔹 ROUGE-based evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
🔹 Graph visualization of performance
🔹 User-friendly interface
🔹 Batch processing of multiple reports
🧠 Tech Stack
Language: Python
NLP Model: T5-small (Hugging Face Transformers)
Framework: PyTorch
Evaluation: ROUGE-score
Visualization: Matplotlib
UI: (Streamlit / CLI — update based on yours)
IDE: VS Code
📂 Project Structure
├── reports/                 # Input medical reports (.txt)
├── summaries/              # Generated summaries
├── medical_summarizer.py   # Main script
├── results.csv             # ROUGE evaluation results
├── rouge_graph.png         # Performance graph
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/medical-report-summarizer.git
cd medical-report-summarizer
2. Install dependencies
pip install -r requirements.txt
▶️ Usage
Run the summarizer:
python medical_summarizer.py
For UI (if Streamlit):
streamlit run app.py
📊 Output
Structured summaries saved in summaries/
Evaluation results stored in results.csv
Graph generated as rouge_graph.png
📈 Evaluation

The system uses ROUGE metrics:

ROUGE-1 → word overlap
ROUGE-2 → phrase overlap
ROUGE-L → sequence similarity

Note: Reference summaries are approximated using diagnosis and treatment sections.

⚠️ Limitations
Uses synthetic / sample medical reports
Not fine-tuned on medical datasets
ROUGE uses approximate reference summaries
Possible hallucination in generated summaries
🔮 Future Improvements
Use medical-specific models (PEGASUS, ClinicalBERT)
Fine-tune on real datasets (MIMIC-IV)
Improve hallucination detection
Add Named Entity Recognition (NER)
Deploy as a web application
