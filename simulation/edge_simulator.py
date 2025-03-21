# simulation/edge_simulator.py
import time
import numpy as np
import argparse

def simulate_edge_processing(duration=10):
    """Basic simulation without model dependency"""
    for t in range(duration):
        print(f"Cycle {t+1}: Simulating sensor data...")
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False)
    args = parser.parse_args()
    
    print("Starting edge simulation...")
    simulate_edge_processing()
