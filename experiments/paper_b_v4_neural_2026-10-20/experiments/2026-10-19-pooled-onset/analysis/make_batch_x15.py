"""Write the X15 batch file for run_batch_steer.py on the VM from the frozen predictions. Usage: python make_batch_x15.py <pred_dir> <out batch.json>
Arms follow prereg/PREREG.md; arm X15_c50_C is included only if X15_c50_A is predicted to oscillate on >= 8 panels."""
import json, sys
import numpy as np
pred_dir, out = sys.argv[1:3]
P = json.load(open(f'{pred_dir}/predictions_x15.json'))
VM = '/home/azureuser/x15'
ids3 = P['panels']['3']['ids']; ids4 = P['panels'].get('4', {}).get('ids', [])
rng = np.random.default_rng(0)
sel3 = sorted(rng.choice(ids3, size=min(64, len(ids3)), replace=False).tolist())
sel4 = sorted(rng.choice(ids4, size=min(64, len(ids4)), replace=False).tolist()) if ids4 else []
common3 = f"--panels {VM}/data/panels3.json --policy /home/azureuser/v4val/models/Qwen2.5-0.5B-Instruct --inner 40 --lr 2e-5 --lam 1.0 --rounds 120 --ids {','.join(map(str, sel3))}"
common4 = f"--panels {VM}/data/panels4.json --policy /home/azureuser/v4val/models/Qwen2.5-0.5B-Instruct --inner 40 --lr 2e-5 --lam 1.0 --rounds 120 --ids {','.join(map(str, sel4))}"
A = '--alpha 0.0 --beta 10.0'; C = '--alpha 0.9 --beta 1.0'
orc = lambda tag: f"--oracle npz:{VM}/oracles/oracle_{tag}.npz"
runs = {
    'X15_c0_A': f"--objective dpo {orc('K3_m1.0_owner_c0')} {A}",
    'X15_c25_A': f"--objective dpo {orc('K3_m1.0_owner_c25')} {A}",
    'X15_c50_A': f"--objective dpo {orc('K3_m1.0_owner_c50')} {A}",
    'X15_c100_A': f"--objective dpo {orc('K3_m1.0_owner_c100')} {A}",
    'X15_c100_C': f"--objective dpo {orc('K3_m1.0_owner_c100')} {C}",
    'X15_rand_A': f"--objective dpo {orc('K3_m1.0_owner_c0')} {A} --sampled 512",
    'X15_c100_A_ipo': f"--objective ipo {orc('K3_m1.0_owner_c100')} {A}",
    'X15_m05_c100_A': f"--objective dpo {orc('K3_m0.5_owner_c100')} {A}",
}
if P['arms']['X15_c50_A']['n_osc'] >= 8:
    runs['X15_c50_C'] = f"--objective dpo {orc('K3_m1.0_owner_c50')} {C}"
batch = {'common': common3, 'runs': runs, 'parallel': 4}
batch4 = {'common': common4, 'runs': {'X15_K4_c100_A': f"--objective dpo {orc('K4_m1.0_owner_c100')} {A}",
                                      'X15_K4_c0_A': f"--objective dpo {orc('K4_m1.0_owner_c0')} {A}"}, 'parallel': 2} if sel4 else None
json.dump(batch, open(out, 'w'), indent=1)
if batch4: json.dump(batch4, open(out.replace('.json', '_K4.json'), 'w'), indent=1)
json.dump(dict(sel3=sel3, sel4=sel4), open(out.replace('.json', '_selection.json'), 'w'))
print('K3 runs', list(runs), 'panels', len(sel3), '| K4 panels', len(sel4))
