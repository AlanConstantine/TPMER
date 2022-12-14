HKU_DIR = r'./HKU956/1. physiological_signals/'  # HKU956 dir
KEC_DIR = r'./KEmoCon/segments/'  # KEmoCon dir

PROCESSED_DIR = r'./processed_signal'

SIGNALS = ['BVP', 'EDA', 'HR', 'TEMP', 'IBI']
SAMPLERATE = {'BVP': 64, 'EDA': 4, 'HR': 1, 'TEMP': 4}
CUTOFF = {'BVP': [1, 8], 'TEMP': [0.005, 0.1], 'EDA': 2}
