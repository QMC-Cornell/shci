#!/usr/bin/env python
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_file', default='result.json')
parser.add_argument('--save_figure', type=bool, default=True)
parser.add_argument('--show_figure', type=bool, default=True)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--exponential', type=bool, default=False)
parser.add_argument('--preprint', type=bool, default=False)
parser.add_argument('--n_points', type=int, default=0)
parser.add_argument('--no_first_order', type=bool, default=False)
parser.add_argument('--pair_contrib', type=bool, default=False)
args = parser.parse_args()

# Read JSON
result = open(args.result_file).read()
result = json.loads(result)

if args.preprint is True:
    plt.figure(figsize=(5.5, 4.0))

# Construct x and y
x = []
y = []
e = []
energy_vars = result['energy_var']
for eps_var, energy_var in energy_vars.items():
    if result['energy_total'].get(eps_var) is None:
        continue
    energy_totals = result['energy_total'][eps_var]
    eps_pt = 1.0
    for eps_pt_iter, energy_total_iter in energy_totals.items():
        eps_pt_iter = float(eps_pt_iter)
        if eps_pt_iter < eps_pt:
            eps_pt = eps_pt_iter
            energy_total = energy_total_iter['value']
            energy_uncert = energy_total_iter['uncert']
    y.append(energy_total)
    x.append(energy_var - energy_total)
    e.append(energy_uncert)
x = np.array(x)
y = np.array(y)
e = np.array(e)

if args.n_points > 0:
    smallest_points = e.argsort()[:args.n_points]
    x = x[smallest_points]
    y = y[smallest_points]

if args.exponential:
    x0 = x[0]
    y_mean= y.mean()
    y_std = y.std()
    x = x / x[0]
    y = (y - y.mean()) / y.std()

# Fit
def model_aug(x):
    x_aug = (x, )
    if args.no_first_order:
        x_aug = ()
    for i in range(2, (args.order + 1)):
        x_aug = x_aug + (x**i, )
    x_aug = np.column_stack(x_aug)
    x_aug = sm.add_constant(x_aug)
    return x_aug

x_aug = model_aug(x)
weights = 1.0 / x**2
alpha = 0.05
point = np.zeros(x_aug.shape[1])
point[0] = 1.0
t = scipy.stats.t.ppf((2 - alpha) / 2., x.shape[0] - 3)
tt = t * 2

def func(x, a, b, c):
    return a + b * np.exp(c * x)

if args.pair_contrib:
    pair_contribs = result['pair_contrib']
    zs = ()
    for eps_var, pair_contrib_filename in pair_contribs.items():
        df = pd.read_csv(pair_contrib_filename)
        df_out = df
        zs = zs + (df['pair_contrib'].values, )
    zs = np.column_stack(zs)
    if args.n_points > 0:
        zs = zs[:, smallest_points]
    df_out['pair_contrib_uncert'] = 0.0
    for i in range(zs.shape[0]):
        fit = sm.WLS(zs[i], x_aug, weights).fit()
        predict = fit.get_prediction(point).summary_frame(alpha=alpha)
        predict = predict.iloc[0]
        energy = fit.params[0]
        uncert = predict['mean_ci_upper'] - predict['mean']
        if args.exponential:
            zs_mean = zs[i].mean()
            zs_std = zs[i].std()
            zs2 = (zs[i] - zs_mean) / zs_std
            succeed = False
            for ii in range(-15, 15, 3):
                for jj in range(-15, 15, 3):
                    try:
                        popt, pcov = curve_fit(func, x, zs2, sigma=1 / weights, p0=[ii, jj, 0])
                        succeed = True
                        break
                    except:
                        pass
                if succeed:
                    break
            if not succeed:
                print('Failed')
                exit(0)
            print(popt)
            energy = (popt[0] + popt[1]) * zs_std + zs_mean
            uncert = np.sqrt(pcov[0][0] + pcov[1][1] + 2 * pcov[0][1]) * zs_std * tt
        df_out['pair_contrib'].values[i] = energy
        df_out['pair_contrib_uncert'].values[i] = uncert
    print(df_out)
    df_out.to_csv('pair_contrib_extrapolate.csv', index=False)

fit = sm.WLS(y, x_aug, weights).fit()
# print(fit.summary())
predict = fit.get_prediction(point).summary_frame(alpha=alpha)
predict = predict.iloc[0]
energy = fit.params[0]
uncert = predict['mean_ci_upper'] - predict['mean']
if args.exponential:
    succeed = False
    for ii in range(-15, 15, 3):
        for jj in range(-15, 15, 3):
            try:
                popt, pcov = curve_fit(func, x, y, sigma=1 / weights, p0=[ii, jj, 0])
                succeed = True
                break
            except:
                pass
        if succeed:
            break
    if not succeed:
        print('Extrapolation failed')
        exit(0)
    energy = (popt[0] + popt[1]) * y_std + y_mean
    t = scipy.stats.t.ppf((2 - alpha) / 2., x.shape[0] - 3)
    uncert = np.sqrt(pcov[0][0] + pcov[1][1] + 2 * pcov[0][1]) * y_std * tt
print('(%.2f Conf.) Extrapolated Energy: %.10f +- %.10f' % ((1.0 - alpha, energy, uncert)))
if np.isnan(uncert):
    uncert = 9999
result['energy_total']['extrapolated'] = {
    'value': energy,
    'uncert': uncert
}
with open(args.result_file, 'w') as result_file:
    json.dump(result, result_file, indent=2)


# Plot
x_fit = np.linspace(0, np.max(x * 1.2), 50)
x_fit_aug = model_aug(x_fit)
if args.exponential:
    y_fit = func(x_fit, *popt)
    x_fit = x_fit * x0
    y_fit = y_fit * y_std + y_mean
    x = x * x0
    y = y * y_std + y_mean
else:
    y_fit = fit.predict(x_fit_aug)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.errorbar(x, y, yerr=e, marker='o', ls='')
plt.plot(x_fit, y_fit, color='grey', ls='--', zorder=0.1)
plt.xlabel('$E_{var} - E_{tot}$ (Ha)')
plt.ylabel('$E_{tot}$ (Ha)')
plt.title('Extrapolation')
plt.xlim(0)
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.grid(True, ls=':')
if args.save_figure:
    plt.savefig('extrapolate.eps')
if args.show_figure:
    plt.show()
