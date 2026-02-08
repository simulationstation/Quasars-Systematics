import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def radec_to_unit(ra_deg: float, dec_deg: float):
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    return np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ], dtype=float)


def ang_sep_deg(ra1, dec1, ra2, dec2):
    u1 = radec_to_unit(ra1, dec1)
    u2 = radec_to_unit(ra2, dec2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def main():
    root = Path(__file__).resolve().parent

    pure_path = Path('outputs/radio_nb_pure_d005_ic_20260208_082028UTC/radio_nb_dipole_audit.json')
    decraexp_path = Path('outputs/radio_nb_decraexp_d005_ic_20260208_082433UTC/radio_nb_dipole_audit.json')
    phys_path = Path('outputs/radio_nb_phys_templates_exp_ic_20260208_082028UTC/radio_nb_dipole_audit.json')
    inj_path = Path('outputs/radio_nb_fast_disproval_20260208_072436UTC/radio_nb_dipole_audit.json')

    for p in [pure_path, decraexp_path, phys_path, inj_path]:
        if not p.exists():
            raise SystemExit(f'missing required input: {p}')

    pure = load_json(str(pure_path))
    decraexp = load_json(str(decraexp_path))
    phys = load_json(str(phys_path))
    inj = load_json(str(inj_path))

    cmb_ra = float(pure['meta']['cmb_ra_deg'])
    cmb_dec = float(pure['meta']['cmb_dec_deg'])

    # ------------------------
    # Fig 1: Direction compare
    # ------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    models = [
        ('Pure', pure, 'tab:blue'),
        ('Exp(Dec/RA)', decraexp, 'tab:orange'),
        ('Exp(+phys)', phys, 'tab:green'),
    ]
    keys = [
        ('RACS+NVSS', 'RACS-low + NVSS', 'o'),
        ('LoTSS+RACS+NVSS', 'LoTSS-DR2 + RACS-low + NVSS', 's'),
    ]

    for label, payload, color in models:
        for klabel, key, marker in keys:
            mp = payload['fits'][key]['map']
            ax.scatter(mp['ra_deg'], mp['dec_deg'], s=60, marker=marker, color=color, edgecolor='k', linewidth=0.5)

    # CMB direction
    ax.scatter(cmb_ra, cmb_dec, s=120, marker='*', color='k', label='CMB dipole')

    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title('Radio Dipole Fits: Direction Shifts Under Nuisance Templates')

    # Legend as custom handles
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='RACS+NVSS', markerfacecolor='gray', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='LoTSS+RACS+NVSS', markerfacecolor='gray', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='tab:blue', label='Pure', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='tab:orange', label='Exp(Dec/RA)', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='tab:green', label='Exp(+phys)', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='*', color='k', label='CMB dipole', markersize=10),
    ]
    ax.legend(handles=legend_items, loc='lower left', frameon=True, fontsize=9)

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / 'fig1_direction_compare.png', dpi=220)
    plt.close(fig)

    # ---------------------------
    # Fig 2: IC / likelihood bars
    # ---------------------------
    def extract_ic(payload, key):
        mp = payload['fits'][key]['map']
        return {
            'loglike': float(mp['loglike']),
            'aic': float(mp['ic']['aic']),
            'bic': float(mp['ic']['bic']),
            'd': float(mp['d']),
            'ra': float(mp['ra_deg']),
            'dec': float(mp['dec_deg']),
        }

    fit_keys = [
        ('RACS+NVSS', 'RACS-low + NVSS'),
        ('LoTSS+RACS+NVSS', 'LoTSS-DR2 + RACS-low + NVSS'),
    ]

    rows = []
    for fit_label, key in fit_keys:
        base = extract_ic(pure, key)
        for model_label, payload, _color in models:
            cur = extract_ic(payload, key)
            rows.append({
                'fit': fit_label,
                'model': model_label,
                'dloglike': cur['loglike'] - base['loglike'],
                'dAIC': cur['aic'] - base['aic'],
                'dBIC': cur['bic'] - base['bic'],
            })

    # Plot grouped bars for dBIC (primary) and dAIC
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    fit_order = [fk[0] for fk in fit_keys]
    model_order = [m[0] for m in models]
    x = np.arange(len(fit_order))
    width = 0.22

    colors = {'Pure': 'tab:blue', 'Exp(Dec/RA)': 'tab:orange', 'Exp(+phys)': 'tab:green'}

    for i, model_label in enumerate(model_order):
        dBIC = []
        dAIC = []
        for fit_label in fit_order:
            r = next(rr for rr in rows if rr['fit'] == fit_label and rr['model'] == model_label)
            dBIC.append(r['dBIC'])
            dAIC.append(r['dAIC'])
        xpos = x + (i - 1) * width
        ax.bar(xpos, dBIC, width=width, color=colors[model_label], alpha=0.8, label=f'{model_label} (dBIC)')
        # draw dAIC as thin line markers
        ax.plot(xpos, dAIC, color='k', linestyle='none', marker='_', markersize=12)

    ax.axhline(0.0, color='k', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(fit_order, rotation=0)
    ax.set_ylabel('Delta IC vs Pure (negative is better)')
    ax.set_title('Model Comparison: Delta BIC (bars) with Delta AIC (black ticks)')
    ax.grid(axis='y', alpha=0.25)

    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    # Keep only unique model labels for bars
    seen = set()
    keep_h = []
    keep_l = []
    for h, l in zip(handles, labels):
        m = l.split(' (')[0]
        if m in seen:
            continue
        seen.add(m)
        keep_h.append(h)
        keep_l.append(m)
    ax.legend(keep_h, keep_l, loc='upper right', frameon=True, fontsize=9, title='Models')

    fig.tight_layout()
    fig.savefig(root / 'fig2_delta_ic.png', dpi=220)
    plt.close(fig)

    # ---------------------------------
    # Fig 3: Template coefficients (tri)
    # ---------------------------------
    tri_key = 'LoTSS-DR2 + RACS-low + NVSS'
    mp = phys['fits'][tri_key]['map']
    names = mp['template_names']
    coefs = mp['template_coef']

    # Order a subset for readability
    order = list(range(len(names)))

    surveys = ['LoTSS-DR2', 'RACS-low', 'NVSS']
    coef_mat = np.array([coefs[s] for s in surveys], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ind = np.arange(len(names))
    w = 0.25
    for i, s in enumerate(surveys):
        ax.bar(ind + (i - 1) * w, coef_mat[i], width=w, label=s)

    ax.axhline(0.0, color='k', linewidth=1.0)
    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Template coefficient (bounded)')
    ax.set_title('Exp(+phys) Joint Fit: Per-survey Template Coefficients')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(root / 'fig3_template_coefficients.png', dpi=220)
    plt.close(fig)

    # ---------------------------------
    # Fig 4: Injection recovery summary
    # ---------------------------------
    inj_sum = inj['injection_recovery']
    pure_s = inj_sum['pure']
    tmpl_s = inj_sum['template_linear']

    def p16p50p84(dct):
        return float(dct['p16']), float(dct['p50']), float(dct['p84'])

    d_p = p16p50p84(pure_s['d'])
    d_t = p16p50p84(tmpl_s['d'])
    a_p = p16p50p84(pure_s['axis_err_deg'])
    a_t = p16p50p84(tmpl_s['axis_err_deg'])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    # axis error
    for ax, (label, vals, color) in zip(axes, [
        ('Recovered d', (d_p, d_t), ('tab:blue', 'tab:orange')),
        ('Axis error (deg)', (a_p, a_t), ('tab:blue', 'tab:orange')),
    ]):
        pass

    # Left: d
    ax = axes[0]
    for i, (lab, v, c) in enumerate([('Pure', d_p, 'tab:blue'), ('Template-linear', d_t, 'tab:orange')]):
        p16, p50, p84 = v
        ax.errorbar(i, p50, yerr=[[p50 - p16], [p84 - p50]], fmt='o', color=c, capsize=4)
        ax.text(i, p84 + 0.002, lab, ha='center', va='bottom', fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel('Recovered dipole amplitude d')
    ax.set_title('Injection recovery (nsim=%d)' % int(inj_sum['nsim']))
    ax.grid(axis='y', alpha=0.25)

    # Right: axis error
    ax = axes[1]
    for i, (lab, v, c) in enumerate([('Pure', a_p, 'tab:blue'), ('Template-linear', a_t, 'tab:orange')]):
        p16, p50, p84 = v
        ax.errorbar(i, p50, yerr=[[p50 - p16], [p84 - p50]], fmt='o', color=c, capsize=4)
        ax.text(i, p84 + 2.0, lab, ha='center', va='bottom', fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel('Angular separation (deg)')
    ax.set_title('Axis recovery error')
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(root / 'fig4_injection_recovery.png', dpi=220)
    plt.close(fig)

    # Write a small machine-readable summary
    summary = {
        'inputs': {
            'pure': str(pure_path),
            'decraexp': str(decraexp_path),
            'phys': str(phys_path),
            'injection': str(inj_path),
        },
        'cmb': {'ra_deg': cmb_ra, 'dec_deg': cmb_dec},
        'model_compare': rows,
        'injection': {
            'nsim': inj_sum['nsim'],
            'pure_axis_err_p50': pure_s['axis_err_deg']['p50'],
            'template_axis_err_p50': tmpl_s['axis_err_deg']['p50'],
        },
    }
    with open(root / 'figure_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
