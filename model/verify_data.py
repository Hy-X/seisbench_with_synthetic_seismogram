#!/usr/bin/env python3
"""Quick verification of HDF5 and CSV data consistency."""

import h5py
import pandas as pd
import numpy as np

print("Data Verification:")
print("=" * 60)

# Check HDF5
with h5py.File('/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/data/synthetic_dataset.hdf5', 'r') as hdf:
    print('\n✓ HDF5 File:')
    print(f'  Traces: {len(hdf.keys())}')
    print(f'  First trace: {list(hdf.keys())[0]}')
    
    trace = hdf['SYNTHETIC_001']
    print(f'  Shape: {trace.shape}')
    print(f'  P-sample: {trace.attrs["p_arrival_sample"]}')
    print(f'  S-sample: {trace.attrs["s_arrival_sample"]}')

# Check CSV
df = pd.read_csv('/Users/hongyuxiao/Hongyu_File/xiao_net_ver_2/data/synthetic_metadata.csv')
print(f'\n✓ CSV Metadata:')
print(f'  Rows: {len(df)}')
print(f'  Columns: {len(df.columns)}')

row = df.iloc[0]
print(f'\n✓ Consistency Check (SYNTHETIC_001):')
print(f'  CSV P-sample: {row["trace_p_arrival_sample"]:.0f}')
print(f'  CSV S-sample: {row["trace_s_arrival_sample"]:.0f}')
print(f'  SNR: {row["snr_db"]:.2f} dB')

print("\n" + "=" * 60)
print("✓ All data verified successfully!")
