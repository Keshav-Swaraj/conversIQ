import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_sample_pdf(filename="data/sample_ai_report.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        spaceAfter=20
    )
    story.append(Paragraph("ConverseIQ Technical Overview & System Architecture", title_style))
    story.append(Spacer(1, 12))
    
    # Section 1
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        leading=20,
        spaceBefore=15,
        spaceAfter=10
    )
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Paragraph(
        "ConverseIQ is an offline-first, voice-enabled document intelligence platform designed for privacy-preserving document retrieval and narration. "
        "It eliminates the need for cloud-based vector databases by leveraging a lightweight, high-precision hierarchical BM25 sparse retriever coupled with a cross-encoder reranker. "
        "The system runs efficiently on standard consumer CPU hardware, delivering high factual accuracy and low latency without transmitting sensitive user data to external cloud services.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    
    # Section 2
    story.append(Paragraph("2. Hierarchical Retrieval Architecture", heading_style))
    story.append(Paragraph(
        "Standard dense retrieval pipelines rely heavily on vector embeddings and high-dimensional indexes, which demand substantial memory and specialized hardware. "
        "In contrast, ConverseIQ adopts a three-tiered hierarchical chunking strategy: Section Level, Paragraph Level, and Sentence Level. "
        "During document ingestion, text extracted from PDFs is split dynamically along semantic section boundaries. "
        "When a user submits a query, BM25 operates across candidate paragraphs to rapidly narrow down the search space to the top 10 most relevant text blocks.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Following sparse candidate retrieval, a cross-encoder reranker based on MiniLM evaluates the top 10 paragraph candidates. "
        "The cross-encoder performs deep cross-attention between the query and candidate passages to compute precise semantic similarity scores. "
        "The top 3 reranked chunks are subsequently formatted into a structured context window and forwarded to the local language model for answer synthesis.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    
    # Section 3
    story.append(Paragraph("3. Local LLM Integration and Memory Management", heading_style))
    story.append(Paragraph(
        "ConverseIQ integrates with Ollama to run local open-weights language models such as Llama 3 and Gemma 2. "
        "To guarantee strict factual grounding and eliminate hallucinations, system prompts enforce a strict constraint: responses must be derived exclusively from the provided document context. "
        "If a question cannot be answered from the document chunks, the model explicitly responds that the information is unavailable in the document.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Conversational state is maintained using a sliding window memory mechanism with a configurable history window k=5. "
        "This allows users to ask multi-turn follow-up questions while preventing prompt overflow and managing local context limits efficiently.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    
    # Section 4
    story.append(Paragraph("4. System Performance and Resource Requirements", heading_style))
    story.append(Paragraph(
        "The average CPU latency for BM25 retrieval across a 50-page document is under 15 milliseconds. "
        "Cross-encoder reranking adds approximately 150 to 300 milliseconds on standard x86 and ARM processors. "
        "Local LLM token generation streams at 15 to 25 tokens per second depending on processor threads, ensuring an interactive and responsive chat experience.",
        styles['Normal']
    ))
    
    doc.build(story)
    print(f"Created sample PDF at {filename}")

if __name__ == "__main__":
    create_sample_pdf()
