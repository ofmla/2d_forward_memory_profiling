import subprocess
import os
import sys
import matplotlib.pyplot as plt


def get_data(command, methods):
    ''' Return data from log file'''
    myDict = {}
    for i in methods:
        word = 'iterate_' + i + '.dat'
        if os.path.isfile(word) is False:
            continue
        process = command[:11] + word + command[11:]
        proc = subprocess.Popen(process, shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout_value = proc.communicate()[0]
        lst = [item for item in stdout_value.decode().split('\n')]
        del lst[-1]
        myDict[i] = [float(j) for j in lst]

    return myDict


def make_a_plot(d, d2, lconfg):
	'''Plot convergence and computational cost curves.'''
    plt.subplots(figsize=(15, 5))
    plt.subplot(1, 2, 1)  # draw plot 1
    for k, v in d.items():
        plt.plot(v, fillstyle='none', linestyle='-', marker=lconfg[k][1],
                 markersize=5, linewidth=1, label=lconfg[k][0])
    plt.xlabel('Iterations')
    plt.ylabel('Relative objective function (log scale)')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', frameon=True, prop={'size': 10})

    plt.subplot(1, 2, 2)  # draw plot 2
    for (k, v), (k2, v2) in zip(d.items(), d2.items()):
        plt.plot(v2, v, fillstyle='none', linestyle='-', marker=lconfg[k][1],
                 markersize=5, linewidth=1, label=lconfg[k][0])
    plt.xlabel('Computed gradients')
    plt.ylabel('Relative objective function (logscale)')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', frameon=True, prop={'size': 10})

    plt.savefig('convergence-cost_curves.pdf')


def main():

    words = ['ST', 'CG', 'LB']
    lconfg = {'ST': ['Steepest descent', 'x'],
              'CG': ['Nonlinear conjugate gradient', '^'],
              'LB': ['l-BFGS', 's']}

    strng = "head -n -3  | awk 'NR >= 10 {print $4}'"
    d = get_data(strng, words)
    if d == {}:
        sys.exit('There are not Log files, '
                 'are you sure you have executed an inversion workflow?')
    strng = "head -n -3  | awk 'NR >= 10 {print $(NF)}'"
    d2 = get_data(strng, words)

	with plt.style.context('dark_background'):
		make_a_plot(d, d2, lconfg)


if __name__ == "__main__":
    main()
