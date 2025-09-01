#!/usr/bin/env python3
"""
Example Python script for SciDoc testing.
"""

import pandas as pd
import numpy as np

def process_data(data):
    """Process the input data."""
    return data * 2

def main():
    """Main function."""
    print("Hello from SciDoc example!")
    data = np.array([1, 2, 3, 4, 5])
    result = process_data(data)
    print(f"Processed data: {result}")

if __name__ == "__main__":
    main()
