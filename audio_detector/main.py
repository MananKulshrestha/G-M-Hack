import argparse
import sys
import os
import random
import time
from colorama import init, Fore, Style

from audio_processor import AudioProcessor
from model_dispatcher import ModelDispatcher
from consensus_engine import ConsensusEngine
from config import MODEL_REGISTRY

# Initialize Colorama
init(autoreset=True)

def print_header():
    print(Fore.CYAN + Style.BRIGHT + "="*60)
    print(Fore.CYAN + Style.BRIGHT + "   Unified Forensic Architecture: Deepfake Audio Detection")
    print(Fore.CYAN + Style.BRIGHT + "   Multi-Model Ensemble via Serverless Inference")
    print(Fore.CYAN + Style.BRIGHT + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection CLI")
    parser.add_argument("input_file", help="Path to the audio file (.wav, .mp3, .flac)")
    parser.add_argument("--exhaustive", action="store_true", help="Process ALL chunks (slow, may hit rate limits)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(Fore.RED + f"[!] Input file not found: {args.input_file}")
        sys.exit(1)

    print_header()

    # 1. Initialize Components
    processor = AudioProcessor()
    dispatcher = ModelDispatcher()
    consensus = ConsensusEngine()

    try:
        # 2. Preprocessing
        print(Fore.YELLOW + f"[*] Preprocessing Audio...")
        chunks = processor.process_file(args.input_file)
        
        if not chunks:
            print(Fore.RED + "[!] No chunks generated. Exiting.")
            sys.exit(1)

        # 3. Smart Sampling Strategy
        # To avoid hitting API rate limits with 60+ chunks, we sample:
        # - The beginning (intro artifacts)
        # - The middle
        # - The end
        # - Random points in between
        total_chunks = len(chunks)
        selected_chunks = []
        
        if args.exhaustive or total_chunks < 10:
            selected_chunks = chunks
            print(f"[*] Mode: Exhaustive (Processing all {total_chunks} chunks)")
        else:
            # Always take first 2 and last 2
            indices = {0, 1, total_chunks-1, total_chunks-2}
            # Add 4 random indices from the middle
            middle_pool = list(range(2, total_chunks-2))
            if middle_pool:
                indices.update(random.sample(middle_pool, min(4, len(middle_pool))))
            
            # Sort indices to process in order
            sorted_indices = sorted(list(indices))
            selected_chunks = [chunks[i] for i in sorted_indices]
            print(f"[*] Mode: Smart Sampling (Processing {len(selected_chunks)} key segments out of {total_chunks})")

        # 4. Model Dispatching
        aggregated_results = {key: 0.0 for key in MODEL_REGISTRY.keys()}
        
        print(Fore.YELLOW + f"[*] Dispatching to Models via API...")
        
        for i, chunk in enumerate(selected_chunks):
            print(f"\r    Analyzing Segment {i+1}/{len(selected_chunks)}...", end="")
            chunk_results = dispatcher.dispatch(chunk)
            
            # Max Pooling Strategy
            for model_key, score in chunk_results.items():
                if score > aggregated_results[model_key]:
                    aggregated_results[model_key] = score
            
            # Tiny sleep to be polite to the API
            time.sleep(0.5)
        
        print("\n" + Fore.GREEN + "[*] Inference Complete.")

        # 5. Consensus & Reporting
        report = consensus.analyze_results(aggregated_results)

        # 6. Display Output
        print("\n" + Fore.WHITE + Style.BRIGHT + "-"*30)
        print(Fore.WHITE + Style.BRIGHT + "       FORENSIC REPORT       ")
        print(Fore.WHITE + Style.BRIGHT + "-"*30)
        
        v_color = Fore.GREEN if "REAL" in report['verdict'] else Fore.RED
        if "INCONCLUSIVE" in report['verdict'] or "UNCERTAIN" in report['verdict']:
            v_color = Fore.YELLOW

        print(f"Final Verdict:      {v_color + Style.BRIGHT}{report['verdict']}")
        print(f"Confidence Level:   {report['confidence']}")
        print(f"CSI Score:          {report['csi']} (Threshold > 2.0 is Strong)")
        print(f"Ensemble Mean:      {report['mean_score']}")
        print(f"Ensemble StdDev:    {report['std_dev']}")
        print("-" * 30)
        print("Model Breakdown (Max Probability):")
        for model, score in report['all_scores'].items():
            model_name = MODEL_REGISTRY[model]['name']
            print(f" - {model_name:<25}: {score:.4f}")
        
        print("-" * 30)
        if report['pessimistic_override']:
             print(Fore.RED + "[!] WARNING: Pessimistic Override Active.")
             print("    At least one model detected a strong artifact despite the average.")

    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user.")
    except Exception as e:
        print(Fore.RED + f"\n[!] Critical System Error: {e}")
    finally:
        # Cleanup
        processor.clean_temp()

if __name__ == "__main__":
    main()