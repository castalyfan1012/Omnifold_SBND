"""
run_sbnd.py — SBND OmniFold training driver.

Renamed from t2k.py to avoid confusion with T2K validation scripts.
Functionally identical to t2k.py but with cleaner diagnostics.

Usage (called by shell scripts and RunStudies.py — do not run directly unless testing):
    python3 run_sbnd.py --config sbnd/config_omnifold_sbnd_closure.json \
        --file_path ../FormattedData_SBND/ \
        --weights_folder weights_sbnd_closure/ \
        --no_eff --verbose
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow.keras as hvd
import tensorflow as tf
import utils
from omnifold import Multifold, LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

parser = argparse.ArgumentParser(description='SBND OmniFold training driver')
parser.add_argument('--config',         default='config_omnifold.json',
                    help='Config file (JSON or Python-dict with single quotes)')
parser.add_argument('--plot_folder',    default='./plots/',
                    help='Folder for plots')
parser.add_argument('--weights_folder', default='./weights/',
                    help='Folder to store output weight files')
parser.add_argument('--file_path',      default='../FormattedData_SBND/',
                    help='Folder containing formatted input .npy files')
parser.add_argument('--nevts',          type=float, default=-1,
                    help='Number of events to use (-1 = all)')
parser.add_argument('--verbose',        action='store_true', default=False,
                    help='Verbose output during training')
parser.add_argument('--shape_only',     action='store_true', default=False,
                    help='Normalise data/MC reco distributions before unfolding')
parser.add_argument('--no_eff',         action='store_true', default=False,
                    help='Omit truth events not reconstructed (use for selected-only dataset)')
flags = parser.parse_args()

nevts = int(flags.nevts)
opt   = LoadJson(flags.config)

# ── Diagnostic printout ───────────────────────────────────────────────────────
print(f"\n=== run_sbnd.py ===")
print(f"  Config:          {flags.config}")
print(f"  Data path:       {flags.file_path}")
print(f"  Weights folder:  {flags.weights_folder}")
print(f"  no_eff:          {flags.no_eff}")
print(f"  NITER:           {opt.get('NITER', '?')}")
print(f"  NTRIAL:          {opt.get('NTRIAL', '?')}  "
      f"({'averaging {0} networks/iter'.format(opt.get('NTRIAL','?'))})")
print(f"  EPOCHS:          {opt.get('EPOCHS', '?')}")
print(f"  BATCH_SIZE:      {opt.get('BATCH_SIZE', '?')}")
print(f"  NAME:            {opt.get('NAME', '?')}")
print(f"==================\n")

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)

data, mc_reco, mc_gen, reco_mask, gen_mask, \
    data_weights, mc_weights, mc_weights_reco = \
    utils.DataLoader(flags.file_path, opt, nevts)

if flags.shape_only:
    mc_weights_reco *= np.sum(data_weights) / np.sum(mc_weights_reco)

if flags.no_eff:
    mc_gen         = mc_gen[gen_mask]
    mc_weights     = mc_weights[gen_mask]
    gen_mask       = gen_mask[gen_mask]

K.clear_session()
mfold = Multifold(
    version='{}'.format(opt['NAME']),
    verbose=flags.verbose,
    config_file=flags.config,
    plot_folder=flags.plot_folder,
    weights_folder=flags.weights_folder,
)
mfold.mc_gen  = mc_gen
mfold.mc_reco = mc_reco
mfold.data    = data
mfold.Preprocessing(
    weights_mc_reco=mc_weights_reco,
    weights_mc=mc_weights,
    weights_data=data_weights,
    pass_reco=reco_mask,
    pass_gen=gen_mask,
)
mfold.Unfold()