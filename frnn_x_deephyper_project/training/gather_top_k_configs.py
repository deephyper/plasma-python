import pandas as pd


path_to_results = "../hps/results/minimalistic-frnn-DBO-async-qUCB-qUCB-16-8-42000-42/results.csv"
k = 80
path_to_configs = f"configs/top_{k}.json"

results = pd.read_csv(path_to_results)

top_k_results = results.nlargest(k, 'objective')
durations = top_k_results['timestamp_end'] - top_k_results['timestamp_start']
objectives = top_k_results['objective']
for i, zipped in enumerate(zip(objectives, durations)):
    obj, t = zipped
    print(f"{i+1}: {obj:.3f} auc in {t:.1f}s.")
top_k_configs = top_k_results.drop(columns=['objective', 'worker_rank', 'timestamp_start', 'timestamp_end', 'Unnamed: 0'])
top_k_configs.to_json(path_to_configs, orient='records')