#!/usr/bin/env python
import argparse
import json
import numpy as np
import pandas as pd
import sys

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_file', default='result.json')
parser.add_argument('--tab', default=False)

args = parser.parse_args()

# Read JSON
result = open(args.result_file).read()
result = json.loads(result)

eps_vars = []
eps_pts = []
energy_vars = []
energy_totals = []
for eps_var, energy_var in result['energy_var'].iteritems():
    eps_vars.append(float(eps_var))
    energy_vars.append(float(energy_var))
    if result['energy_total'].get(eps_var) is not None:
        eps_pt_min = 1.0
        for eps_pt, energy_total in result['energy_total'][eps_var].iteritems():
            eps_pt = float(eps_pt)
            if eps_pt < eps_pt_min:
                eps_pt_min = eps_pt
                energy_total_min = energy_total['value']
        energy_totals.append(float(energy_total_min))
        eps_pts.append(eps_pt_min)
    else:
        energy_totals.append(float('nan'))
        eps_pts.append(float('nan'))

# Output CSV
df = pd.DataFrame({
    'eps_var': np.array(eps_vars),
    'eps_pt': np.array(eps_pts),
    'energy_var': np.array(energy_vars),
    'energy_total': np.array(energy_totals)
}, columns=['eps_var', 'eps_pt', 'energy_var', 'energy_total'])
df.sort_values(by=['eps_var'], ascending=False, inplace=True)
sep = '\t' if args.tab else ','
df.to_csv(sys.stdout, sep=sep, index=False, float_format='%.9f')
