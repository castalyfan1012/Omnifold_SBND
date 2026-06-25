"""
RunSystematicUniverses.py

Runs OmniFold once per systematic universe (BNB flux + GENIE).
Each universe produces a different set of push weights, which are
used to build the covariance matrix of the unfolded result.

Usage:
    python3 sbnd/RunSystematicUniverses.py --source bnb --start 0 --end 100
    python3 sbnd/RunSystematicUniverses.py --source genie --start 0 --end 100
"""


import numpy as np
import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=['bnb', 'genie', 'mcstat'], required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--universe-file', type=str, default=None)
flags = parser.parse_args()

if flags.universe_file is None:
    flags.universe_file = f'sbnd/exported_weights/{flags.source}_universe_weights.npy'

uni_all = np.load(flags.universe_file)   # (6399, 100)
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

for idx in range(flags.start, flags.end):
    tag = f'{flags.source}_univ{idx}'

    base_weights_dir = f'sbnd/weights_{flags.source}'
    # base_plots_dir   = f'sbnd/plots_{flags.source}'

    weights_dir = f'{base_weights_dir}/{tag}/'
    # plots_dir   = f'{base_plots_dir}/{tag}/'

    out_file = flags.data_dir + f'data_weights_sbnd_syst_{tag}.npy'

    # Build fake data weights for this universe
    uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)
    data_weights = mc_weights * uni_vals
    data_weights = data_weights * (mc_weights.sum() / data_weights.sum())
    np.save(out_file, data_weights)

    # Write config
    config = {
        'FILE_MC_RECO':        'mc_vals_reco.npy',
        'FILE_MC_GEN':         'mc_vals_truth.npy',
        'FILE_MC_FLAG_RECO':   'mc_pass_reco.npy',
        'FILE_MC_FLAG_GEN':    'mc_pass_truth.npy',
        'FILE_DATA_RECO':      'mc_vals_reco.npy',
        'FILE_DATA_FLAG_RECO': 'mc_pass_reco.npy',
        'FILE_DATA_WEIGHT':    f'data_weights_sbnd_syst_{tag}.npy',
        'FILE_MC_RECO_WEIGHT': 'mc_weights_reco.npy',
        'FILE_MC_GEN_WEIGHT':  'mc_weights_truth.npy',
        'NITER': 5,        # fewer iterations for systematics (convergence is fast)
        'NTRIAL': 1,       # single trial to save time (100 universes provides averaging)
        'LR': 1e-3,
        'BATCH_SIZE': 512,
        'EPOCHS': 50,
        'NAME': f'sbnd_syst_{tag}',
        'NPATIENCE': 7,
    }

    config_path = f'sbnd/config_syst_{tag}.json'
    # Write as Python-parseable format (matching the repo's existing style)
    with open(config_path, 'w') as f:
        f.write('{\n')
        for i, (k, v) in enumerate(config.items()):
            comma = ',' if i < len(config) - 1 else ''
            if isinstance(v, str):
                f.write(f"'{k}':'{v}'{comma}\n")
            else:
                f.write(f"'{k}': {v}{comma}\n")
        f.write('}\n')

    os.makedirs(weights_dir, exist_ok=True)
    # os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running universe {idx} ({flags.source})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, 't2k.py',
        '--config', config_path,
        '--file_path', flags.data_dir,
        '--weights_folder', weights_dir,
        # '--plot_folder', plots_dir,
        '--no_eff', '--verbose'
    ]
    subprocess.run(cmd)

    # Clean up config and intermediate data weight files
    os.remove(config_path)

print("\nAll universes complete.")