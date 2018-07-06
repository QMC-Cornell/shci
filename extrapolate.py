#!/usr/bin/env python
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_file', default='result.json')
parser.add_argument('--save_figure', type=bool, default=True)
parser.add_argument('--show_figure', type=bool, default=True)
args = parser.parse_args()

# Read JSON
result = open(args.result_file).read()
result = json.loads(result)

# Construct x and y
x = []
y = []
energy_vars = result['energy_var']
for eps_var, energy_var in energy_vars.iteritems():
    if result['energy_total'].get(eps_var) is None:
        continue
    energy_totals = result['energy_total'][eps_var]
    eps_pt = 1.0
    for eps_pt_iter, energy_total_iter in energy_totals.iteritems():
        eps_pt_iter = float(eps_pt_iter)
        if eps_pt_iter < eps_pt:
            eps_pt = eps_pt_iter
            energy_total = energy_total_iter['value']
    y.append(energy_total)
    x.append(energy_var - energy_total)

# Fit
def model_aug(x):
    x_aug = np.column_stack((x, x**2))
    x_aug = sm.add_constant(x_aug)
    return x_aug

x = np.array(x)
y = np.array(y)
x_aug = model_aug(x)
weights = 1.0 / x**2
print(x)
print(weights)
fit = sm.WLS(y, x_aug, weights).fit()
print(fit.summary())
alpha = 0.05
predict = fit.get_prediction(np.array([1.0, 0.0, 0.0])).summary_frame(alpha=alpha)
print(predict)
predict = predict.iloc[0]
energy = fit.params[0]
uncert = predict['mean_ci_upper'] - predict['mean']
print('(%.2f Conf.) Extrapolated Energy: %.10f +- %.10f' % ((1.0 - alpha, fit.params[0], uncert)))
result['energy_total']['extrapolated'] = {
    'value': energy,
    'uncert': uncert
}
with open(args.result_file, 'w') as result_file:
    json.dump(result, result_file, indent=2)


# Plot
x_fit = np.linspace(0, np.max(x * 1.2), 50)
x_fit_aug = model_aug(x_fit)
y_fit = fit.predict(x_fit_aug)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
plt.plot(x, y, marker='o', ls='')
plt.plot(x_fit, y_fit, color='grey', ls='--', zorder=0.1)
plt.xlabel('$E_{var} - E_{tot}$ (Ha)')
plt.ylabel('$E_{tot}$ (Ha)')
plt.title('Weighted Quadratic Extrapolation')
plt.xlim(0)
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.grid(True, ls=':')
if args.save_figure:
    plt.savefig('extrapolate.eps')
if args.show_figure:
    plt.show()
