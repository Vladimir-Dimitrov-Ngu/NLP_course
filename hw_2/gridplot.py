import matplotlib.pyplot as plt
import pandas as pd

def plot_grid(grid_search):
  plt.rcParams["figure.figsize"] = (14, 12)
  results = pd.DataFrame(grid_search.cv_results_)
  results["params_str"] = results.params.apply(str)
  results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
  mean_scores = results.pivot(
      index="iter",
      columns="params_str",
      values="mean_test_score",
  )
  ax = mean_scores.plot(legend=False, alpha=0.6)

  labels = [
      f"iter={i}\nn_samples={grid_search.n_resources_[i]}\nn_candidates={grid_search.n_candidates_[i]}"
      for i in range(grid_search.n_iterations_)
  ]

  ax.set_xticks(range(grid_search.n_iterations_))
  ax.set_xticklabels(labels, rotation=45, multialignment="left")
  ax.set_title("Scores of candidates over iterations")
  ax.set_ylabel("Mean test score", fontsize=15)
  ax.set_xlabel("Iterations", fontsize=15)
  plt.tight_layout()
  plt.grid()
  plt.show()