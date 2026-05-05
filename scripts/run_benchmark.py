import subprocess
import itertools
import sys

def main():
    graph_types = ['retweet', 'reply', 'mention', 'late_fuse']
    encoders = ['gcn', 'gat', 'gatv2', 'gine', 'sage', 'transformer', 'nnconv']
    
    print(f"Starting Benchmark Suite for GNN Link Prediction")
    print(f"Total configurations to test: {len(graph_types) * len(encoders)}")
    
    for graph_type in graph_types:
        for encoder in encoders:
            print(f"\n{'='*80}")
            print(f"Running Experiment: Graph={graph_type.upper()} | Encoder={encoder.upper()}")
            print(f"{'='*80}")
            
            # Use sys.executable to ensure we use the same Python interpreter
            cmd = [
                sys.executable,
                "Main.py",
                "--graph_type", graph_type,
                "--encoder", encoder
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Experiment failed for Graph={graph_type}, Encoder={encoder}")
                print(e)
                continue
                
    print("\nBenchmark Suite Completed successfully.")

if __name__ == "__main__":
    main()
