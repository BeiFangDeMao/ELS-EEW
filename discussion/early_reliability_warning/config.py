"""Fixed experiment configuration, paths relative to the existing workspace."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE/'els_eew'
EXPERIMENT = PROJECT/'outputs/experiment_20261012_223631_490021/CompactGRU_seed42'
CHECKPOINT_DIR = EXPERIMENT/'checkpoint_CompactGRU_20261012_235806'
WEIGHTS = CHECKPOINT_DIR/'best_CompactGRU_20261012_235806.pth'
SOURCE = WORKSPACE/'dataset/dataseth5/J_K_KIKnet_2976_Causal_100Hz_20261012_153132_099924/J_K_KIKnet_2976_Causal_100Hz.h5'
CACHE = WORKSPACE/'plotting/Fig01_Data_Distribution/labels.npz'
LENGTH,CHUNK,RATE = 2976,32,100
BATCH_SIZE,WORKERS = 256,4
CI_METHOD = '95% Wilson binomial-proportion interval, waveform records as units'
TOLERANCES = {'magnitude':[.1,.2,.3,.5], 'distance_km':[5,10,20,30,50],
              'azimuth_deg':[15,30,45,60,90], 'location_km':[5,10,20,30,50]}
REPORT_TIMES = [1,2,3,5,10,20]
