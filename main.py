"""ConverseIQ Phase 1 Entry Point CLI - Performance Optimized with GPU Acceleration & Profiling."""

import os
import sys
import argparse

# Reconfigure stdout for Windows console UTF-8 support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from core.pipeline import ConverseIQPipeline
from scripts.create_sample_pdf import create_sample_pdf


def main():
    parser = argparse.ArgumentParser(description="ConverseIQ Phase 1 - Voice-First Vectorless RAG Pipeline CLI")
    parser.add_argument("--pdf", type=str, default="data/sample_ai_report.pdf", help="Path to input PDF document")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"Creating sample PDF at '{pdf_path}'...")
        create_sample_pdf(pdf_path)

    print(f"\n=======================================================")
    print(f"[+] ConverseIQ Phase 1 GPU-Accelerated Pipeline Initialized")
    print(f"[+] Target Document: {pdf_path}")
    print(f"=======================================================\n")

    pipeline = ConverseIQPipeline()
    
    # 1. Load Document Session once (caches PDF ingestion & BM25 index)
    session = pipeline.load_document(pdf_path)
    print(f"[+] Document Session Loaded: {len(session.chunks)} hierarchical chunks created in {session.ingestion_timings.get('total_ingestion_ms', 0):.2f} ms.")
    print("[+] Ready for interactive Q&A! Type 'exit' to quit, 'clear' to reset chat memory.\n")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nUser Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Exiting ConverseIQ Session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q", "/bye"]:
            print("[+] Exiting ConverseIQ Session. Goodbye!")
            break

        if user_input.lower() in ["clear", "reset"]:
            pipeline.clear_memory()
            print("[+] Conversation memory cleared.")
            continue

        print("\nConverseIQ Answer: ", end="", flush=True)
        tokens = []
        for token in pipeline.stream_query(user_input):
            tokens.append(token)
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()

