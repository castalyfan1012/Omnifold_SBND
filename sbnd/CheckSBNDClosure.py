# run as: python3 CheckSBNDClosure.py
import numpy as np, glob, re

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

push_files = sorted(glob.glob('weights_sbnd_closure/Step2_Iter*_sbnd_nueCC_closure_PushWeights.npy'), key=iter_num)
pull_files = sorted(glob.glob('weights_sbnd_closure/Step1_Iter*_sbnd_nueCC_closure_PullWeights.npy'), key=iter_num)

push = np.load(push_files[-1])
pull = np.load(pull_files[-1])

print(f"Push weights: mean={push.mean():.4f}, std={push.std():.4f}")
print(f"Pull weights: mean={pull.mean():.4f}, std={pull.std():.4f}")
print(f"Push range:   [{push.min():.4f}, {push.max():.4f}]")

# Pass: mean ~1.0, std < 0.1
if abs(push.mean() - 1.0) < 0.05 and push.std() < 0.2:
    print("CLOSURE TEST PASSED — ready for fake data study")
else:
    print("WARNING — weights not centered on 1.0, investigate before proceeding")