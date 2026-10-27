"""Equal-prompt Hodge averages in the original candidate-index coordinates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hodge_diagnostics'))
from hodge.core import flow, hodge


def checked_hodge(a):
    """Check the existing core against an independent edge least-squares fit."""
    part = hodge(a)
    k = len(a)
    i, j = np.triu_indices(k, 1)
    incidence = np.eye(k)[i] - np.eye(k)[j]
    potential = np.linalg.lstsq(incidence, a[i, j], rcond=None)[0]
    error = {
        'skew': float(np.max(np.abs(a + a.T))),
        'cyclic_row_sum': float(np.max(np.abs(part.C.sum(axis=1)))),
        'independent_edge_projection': float(np.max(np.abs(
            part.C[i, j] - (a[i, j] - incidence @ potential)))),
        'orthogonality': float(abs(np.sum(part.G * part.C))),
        'energy_identity': float(abs(np.sum(a*a) - np.sum(part.G**2) - np.sum(part.C**2))),
        'reconstruction': float(np.max(np.abs(a - part.G - part.C))),
    }
    # Near-boundary probabilities amplify floating-point reciprocity errors in logit.
    # Preserve the input and report the discrepancy, without clipping or symmetrizing.
    if max(error.values()) > 1e-6:
        raise ValueError(f'Hodge validation failed: {error}')
    return part, error


def describe(part):
    energy = float(np.sum(part.A**2))
    return {
        'A': part.A.tolist(), 'u': part.u.tolist(),
        'G': part.G.tolist(), 'C': part.C.tolist(),
        'cyclic_spectral_norm': float(np.linalg.norm(part.C, 2)),
        'cyclic_frobenius_norm': float(np.linalg.norm(part.C)),
        'cyclic_energy_share': float(np.sum(part.C**2) / energy) if energy else None,
    }


def aggregate(matrices):
    p = np.asarray(matrices, dtype=float)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[1] != p.shape[2]:
        raise ValueError('Expected nonempty N x K x K matrices')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('All probabilities must be finite and in [0,1]')
    if (not np.allclose(p + p.transpose(0, 2, 1), 1, atol=1e-12, rtol=0)
            or not np.allclose(np.diagonal(p, axis1=1, axis2=2), .5, atol=1e-12, rtol=0)):
        raise ValueError('Expected reciprocal complete matrices with diagonal .5')
    mean_p = p.mean(axis=0)
    result = {'count': len(p), 'K': p.shape[1], 'mean_P': mean_p.tolist(), 'links': {}}
    for link in ('identity', 'logit'):
        entry = {}
        if link != 'logit' or np.all((mean_p > 0) & (mean_p < 1)):
            mean_probability_part, mean_probability_error = checked_hodge(flow(mean_p, link))
            entry.update(hodge_of_link_of_mean_P=describe(mean_probability_part),
                         mean_P_validation_errors=mean_probability_error)
        else:
            entry['mean_P_status'] = 'UNDEFINED_UNCLIPPED_LOGIT'
        endpoints = int(np.sum((p <= 0) | (p >= 1)))
        if link == 'logit' and endpoints:
            entry.update(status='UNDEFINED_UNCLIPPED_PER_PROMPT_LOGIT', endpoint_entries=endpoints)
            result['links'][link] = entry
            continue
        parts, errors = zip(*(checked_hodge(flow(panel, link)) for panel in p))
        a = np.array([part.A for part in parts])
        c = np.array([part.C for part in parts])
        norms = np.linalg.norm(c, ord=2, axis=(1, 2))
        energies = np.sum(a*a, axis=(1, 2))
        c_energies = np.sum(c*c, axis=(1, 2))
        average_part, average_error = checked_hodge(a.mean(axis=0))
        linearity_error = float(np.max(np.abs(average_part.C - c.mean(axis=0))))
        if linearity_error > 1e-12:
            raise ValueError('Hodge/average linearity check failed')
        mean_norm = float(norms.mean())
        rms_norm = float(np.sqrt(np.mean(norms**2)))
        average_norm = float(np.linalg.norm(average_part.C, 2))
        entry.update(
            status='COMPLETE_NO_CLIPPING',
            hodge_of_mean_link_flow=describe(average_part),
            mean_per_prompt_cyclic_spectral_norm=mean_norm,
            rms_per_prompt_cyclic_spectral_norm=rms_norm,
            mean_cyclic_norm_over_mean_norm=average_norm / mean_norm if mean_norm else None,
            mean_cyclic_norm_over_rms_norm=average_norm / rms_norm if rms_norm else None,
            mean_per_prompt_cyclic_energy_share=(float(np.mean(c_energies / energies))
                                                if np.all(energies > 0) else None),
            pooled_cyclic_energy_share=(float(c_energies.sum() / energies.sum())
                                       if energies.sum() else None),
            per_prompt=[dict(cyclic_spectral_norm=float(n), cyclic_energy_share=float(ce/e)
                             if e else None) for n, ce, e in zip(norms, c_energies, energies)],
            max_per_prompt_validation_errors={key: max(e[key] for e in errors) for key in errors[0]},
            mean_flow_validation_errors=average_error,
            linearity_max_error=linearity_error,
        )
        result['links'][link] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.metadata.read_text(encoding='utf-8'))
    if len({row['prompt_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate prompt IDs')
    result = aggregate([row['preference_matrix'] for row in rows])
    result.update(metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                  metadata_canonical_json_sha256=hashlib.sha256(
                      json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                  ).hexdigest(),
                  group_counts=dict(sorted(Counter(row['group'] for row in rows).items())),
                  weighting='equal prompts; complete uniformly weighted edges',
                  coordinates='original candidate indices, not shared semantic labels',
                  norm_convention='matrix spectral L2; Frobenius reported separately',
                  core_sha256=hashlib.sha256(
                      (Path(__file__).resolve().parents[1] / 'hodge_diagnostics/hodge/core.py').read_bytes()
                  ).hexdigest())
    for entry in result['links'].values():
        for source, value in zip(rows, entry.get('per_prompt', [])):
            value.update(prompt_id=source['prompt_id'], group=source['group'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for link, entry in result['links'].items():
        print(link, entry['status'])
        if 'hodge_of_mean_link_flow' in entry:
            print('norm(mean C) =', entry['hodge_of_mean_link_flow']['cyclic_spectral_norm'])
            print('mean(norm C) =', entry['mean_per_prompt_cyclic_spectral_norm'])
            print('aggregate cyclic energy share =', entry['hodge_of_mean_link_flow']['cyclic_energy_share'])


if __name__ == '__main__':
    main()
