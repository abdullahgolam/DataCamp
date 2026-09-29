from pypdf import PdfReader
from transformers import pipeline

# Load the PDF file
reader = PdfReader("courses/03_working_with_hugging_face/docs/chapter2.pdf")

document_text = ""
for page in reader.pages:
    document_text += page.extract_text()

# Load the QA pipeline
qa_pipeline = pipeline(
    model="distilbert-base-cased-distilled-squad")

question = "What is the purpose of this course?"

# Generate an answer
result = qa_pipeline(question=question, context=document_text)
print(f"Answer: {result['answer']}")