import json
import sys

from matplotlib import pyplot as plt


def main(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    keys = sorted(data.keys())

    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot([data[k] for k in keys],
                labels=keys,
                showfliers=False,
                whis='range')

    for i, k in enumerate(keys):
        plt.scatter([i+1 for _ in data[k]], data[k], alpha=0.2, marker='.')

    plt.xticks(fontsize=7, rotation=10)
    plt.ylabel('final Bayes factor')
    plt.title('OED - real data simulation on synthetic data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
