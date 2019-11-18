import json
import sys
from collections import defaultdict

from matplotlib import pyplot as plt


def main(fn):
    with open(fn, 'r') as f:
        data = json.load(f)

    d = defaultdict(list)
    for name, values in data.items():
        for (M, bayes_factor) in values:
            d[(M, name)].append(bayes_factor)

    def adjust_name(name):
        if name == 'oed_alt':
            return 'alt'
        else:
            return name

    plt.figure(figsize=(6.5, 4.5))
    keys = [(M, name) for M, name in sorted(d.keys())]
    plt.boxplot([d[k] for k in keys],
                labels=['{}\nM={}'.format(adjust_name(name), M) for M, name in keys],
                showfliers=False,
                whis='range')

    colours = {'oed': 'r', 'oed_alt': 'g', 'rand': 'b'}
    for i, k in enumerate(keys):
        M, name = k
        plt.scatter([i+1 for _ in d[k]], d[k], c=colours[name], alpha=0.2, marker='.')

    plt.xticks(fontsize=8)
    plt.ylabel('final Bayes factor')
    plt.title('OED - real data simulation on synthetic data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
