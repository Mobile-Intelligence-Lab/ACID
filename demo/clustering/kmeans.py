from __future__ import print_function
import numpy as np
import argparse
import sys
import os

root_path = os.path.dirname(os.path.realpath(__file__)) + '/../..'
sys.path.append(root_path)

from sklearn.cluster import KMeans
from core.utils.misc import Dot

from utils import gen_5_circles, get_blobs, gen_cos_sin

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
plt.ion()

n_samples = 150


if __name__ == "__main__":
    parser = argparse.ArgumentParser('k-Means',
                                     description='Clustering experiments for the paper: "'
                                     'Adaptive Clustering-based Malicious Traffic Classification'
                                     ' at the Network Edge"')
    parser.add_argument('--2-circles', dest='two_circles', action='store_true', help='Include 2-circles dataset')
    parser.add_argument('--5-circles', dest='five_circles', action='store_true', help='Include 5-circles dataset')
    parser.add_argument('--2-moons', dest='two_moons', action='store_true', help='Include 2-moons dataset')
    parser.add_argument('--blobs', dest='blobs', action='store_true', help='Include blobs dataset')
    parser.add_argument('--sine-cosine', dest='sine_cosine', action='store_true', help='Include Sine/Cosine` dataset')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='Save the figures generated')
    args = parser.parse_args()

    SAVE_PLOTS = args.save_plots

    save_dir = f"{root_path}/demo/clustering/plots/kmeans"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    colors = ['b', 'r', 'g', 'm', 'k', 'c', 'c', 'c']

    fig_size = (10, 5.5)
    title_font_size, subplot_title_font_size = 35, 20

    scenarios = {
        "2 Circles": {
            "exclude": not args.two_circles,
            "fig_size": fig_size,
            "font_sizes": {"figure_title": title_font_size, "subplot_title": subplot_title_font_size},
            "n_classes": 2,
            "data": datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05),
            "file_name": "2_circles.pdf"
        },
        "5 Circles": {
            "exclude": not args.five_circles,
            "fig_size": fig_size,
            "font_sizes": {"figure_title": title_font_size, "subplot_title": subplot_title_font_size},
            "n_classes": 5,
            "data": gen_5_circles(),
            "file_name": "5_circles.pdf"
        },
        "2 Moons": {
            "exclude": not args.two_moons,
            "fig_size": fig_size,
            "font_sizes": {"figure_title": title_font_size, "subplot_title": subplot_title_font_size},
            "n_classes": 2,
            "data": datasets.make_moons(n_samples=n_samples, noise=.05),
            "file_name": "2_moons.pdf"
        },
        "Blobs": {
            "exclude": not args.blobs,
            "fig_size": fig_size,
            "font_sizes": {"figure_title": title_font_size, "subplot_title": subplot_title_font_size},
            "n_classes": 3,
            "data": get_blobs(),
            "file_name": "blobs.pdf"
        },
        "Sine / Cosine": {
            "exclude": not args.sine_cosine,
            "fig_size": (14, 4.5),
            "font_sizes": {"figure_title": title_font_size, "subplot_title": 18},
            "n_classes": 2,
            "data": gen_cos_sin(),
            "file_name": "sine_cosine.pdf"
        }
    }

    excluded_scenarios = [key for key in scenarios.keys() if scenarios[key]["exclude"]]
    included_scenarios = list(set(list(scenarios.keys())) - set(excluded_scenarios))

    if len(included_scenarios) == 0:
        included_scenarios = excluded_scenarios

    print(f"Running k-Means on: {included_scenarios}")

    for key in included_scenarios:
        scenario = Dot(scenarios[key])

        _fig, _axes = plt.subplots(1, 2, figsize=scenario.fig_size)

        font_sizes = Dot(scenario.font_sizes)
        _fig.suptitle(key, fontsize=font_sizes.figure_title)
        _axes[0].set_title('Original Data\n', fontsize=font_sizes.subplot_title)
        _axes[1].set_title('Clustering Result\n', fontsize=font_sizes.subplot_title)

        print(f"\nClustering: {key}")
        x, original_labels = scenario.data
        x, original_labels = np.array(x), np.array(original_labels)
        kmeans = KMeans(n_clusters=scenario.n_classes).fit(x)
        labels = kmeans.labels_
        _axes[0].scatter(x[:, 0], x[:, 1], c=[colors[l] for l in original_labels], s=50)
        _axes[1].scatter(x[:, 0], x[:, 1], c=[colors[l] for l in labels], s=50)
        _fig.tight_layout()

        if SAVE_PLOTS:
            print(f"Saving plot for \"{key}\"...")
            _fig.savefig(f"{save_dir}/{scenario.file_name}", bbox_inches="tight", dpi=300)
            print("Plot saved.")

    plt.ioff()
    plt.show()
