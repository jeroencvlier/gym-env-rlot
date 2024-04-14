import os
import wandb

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

from gym_env_rlot.buy_sell.gym_env_v0 import BuySellUndEnv
import shap
import warnings
import torch
from icecream import ic
import json
import torch
import numpy as np
import pandas as pd

from shap.explainers._deep import DeepExplainer as OriginalDeepExplainer
from shap.explainers._deep.deep_pytorch import PyTorchDeep

from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

run = wandb.init()
model_artifact = run.use_artifact(
   "jeroencvlier/rlot-PPO-pipeline-v1-noise-sliceddown-stats-v6-gym-v0/iteration_154:v0",
    type="model",
)
data_artifact = run.use_artifact(model_artifact.metadata["data_artifact"], type="data")
model_path = model_artifact.download()
data_path = data_artifact.download()
wandb.finish()
model_artifact.metadata

with open(f"{model_path}/PPO/gym_env_info.json") as f:
    config = json.load(f)

feature_names = config["feature_names"]


def env_creator(env_config):
    return BuySellUndEnv(env_config)


# check if a ray instance is already running
if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True)

register_env("buysellmulti_env", lambda config: env_creator(config))
algo = PPO.from_checkpoint(os.path.join(model_path, "PPO"))


def shap_evaluation(data_path, algo, test_train="test", num_episodes=1):
    """Evaluate the agent for a given number of episodes."""
    state_log = []
    action_log = []
    env = env_creator({"artifact_path": data_path, "test_train": test_train})
    for _ in range(num_episodes):
        episode_reward = 0
        done = False
        obs, _ = env.reset()
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            state_log.append(obs)
            action_log.append(action)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

    return np.array(state_log), np.array(action_log)


class ModelWrapper(torch.nn.Module):
    def __init__(self, actual_model):
        super().__init__()
        self.actual_model = actual_model

    def forward(self, data):
        input_dict = {"obs_flat": data}
        self.actual_model.eval()

        model_out, _ = self.actual_model.forward(
            input_dict=input_dict, state=[], seq_lens=None
        )
        return model_out


class CustomDeepExplainer:
    def __init__(self, model, data):
        self.model_wrapper = ModelWrapper(model)
        self.data = data
        self.explainer = PyTorchDeep(self.model_wrapper, data)
        self.expected_value = self.explainer.expected_value
        self.framework = "pytorch"

    def shap_values(self, X):
        return self.explainer.shap_values(X)


background_state_log, _ = shap_evaluation(
    data_path, algo, test_train="all", num_episodes=1
)
state_log, action_log = shap_evaluation(data_path, algo, num_episodes=1)


states_tesnsor = torch.tensor(state_log, dtype=torch.float32, requires_grad=True)

background_tensor = torch.tensor(
    background_state_log, dtype=torch.float32, requires_grad=True
)

model = algo.get_policy().model

exp = CustomDeepExplainer(model, background_tensor)
shap_values = exp.shap_values(states_tesnsor)


shap.summary_plot(shap_values, state_log, feature_names, plot_type="bar")
shap.summary_plot(
    shap_values[0], state_log, feature_names, plot_type="violin", max_display=100
)

# # state_log_re = state_log*(env.state_max - env.state_min) + env.state_min
# cmap = "coolwarm"
# fig = plt.figure(figsize=(15, 12))
# gs = fig.add_gridspec(9, hspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# # get pl position in feature names
# pl_pos = feature_names.index("pl")
# intrade_pos = feature_names.index("in_trade")

# axs[0].plot(state_log[:, pl_pos])
# axs[0].plot(state_log[:, pl_pos])


# axs[1].scatter(range(0, len(action_log)), action_log, cmap=cmap)
# axs[1].set_ylabel("action")
# axs[1].yaxis.set_label_position("right")


# axs[3].scatter(
#     range(0, len(shap_values[0][:, 0])),
#     state_log[:, intrade_pos],
#     cmap=cmap,
#     c=shap_values[0][:, intrade_pos],
# )
# axs[3].set_ylabel(feature_names[intrade_pos])
# axs[3].yaxis.set_label_position("right")

# # show
# plt.show()


mean_abs_shap_values = np.mean(np.abs(shap_values[0]), axis=0)
feature_importances = sorted(zip(mean_abs_shap_values, feature_names), reverse=True)


def plot_stats(feature_importances, stat_name):
    plot_df = pd.DataFrame(feature_importances, columns=["importance", "feature"])
    reference_df = pd.DataFrame({"MEAN":plot_df['importance'].mean(),
                                        "MAX":plot_df['importance'].max()} ,index=[0]).T.reset_index().rename(columns={0:'importance','index':'feature'})

    # slice rows that contian mode str
    plot_df_slice = plot_df[plot_df["feature"].str.contains(stat_name)]
    s = plot_df_slice['importance'].sum()
    ic(stat_name,s )
    plot_df_slice = pd.concat([reference_df, plot_df_slice])
    fig, ax = plt.subplots(figsize=(15, 20))
    ax.barh(plot_df_slice["feature"], plot_df_slice["importance"])
    plt.show()
    
    return 

plot_stats(feature_importances, "Q1")
plot_stats(feature_importances, "Q3")
plot_stats(feature_importances, "IQR")
plot_stats(feature_importances, "Mode1")
plot_stats(feature_importances, "Mode2")
plot_stats(feature_importances, "Mode3")
plot_stats(feature_importances, "Mode4")
plot_stats(feature_importances, "Mode5")
plot_stats(feature_importances, "Med")
plot_stats(feature_importances, "Mean")
plot_stats(feature_importances, "Std")
plot_stats(feature_importances, "Range")
plot_stats(feature_importances, "Skew")

plot_impoertance_sum = []
for stat in ['Q1','IQR','Med','Mean','Std','Range','Skew']:
    plot_df = pd.DataFrame(feature_importances, columns=["importance", "feature"])
    plot_df_slice = plot_df[plot_df["feature"].str.contains(stat)]
    s = plot_df_slice['importance'].sum()
    plot_impoertance_sum.append({'feature':stat,'importance':s})
    
plot_impoertance_sum = pd.DataFrame(plot_impoertance_sum)
plot_impoertance_sum.sort_values('importance',ascending=False,inplace=True)
plot_impoertance_sum.plot(kind='bar',x='feature',y='importance',figsize=(12,10))

plot_stats(feature_importances, "_tV_")

plot_stats(feature_importances, "oI")
plot_stats(feature_importances, "tV")

plot_impoertance_sum = []
for stat in ['_tV_','_oI_','_bS_','_aS_' ]:
    plot_df = pd.DataFrame(feature_importances, columns=["importance", "feature"])
    plot_df_slice = plot_df[plot_df["feature"].str.contains(stat)]
    s = plot_df_slice['importance'].sum()
    plot_impoertance_sum.append({'feature':stat,'importance':s})
    
plot_impoertance_sum = pd.DataFrame(plot_impoertance_sum)
plot_impoertance_sum.sort_values('importance',ascending=False,inplace=True)
plot_impoertance_sum.plot(kind='bar',x='feature',y='importance',figsize=(12,10))


    






# Assuming you decide to remove features in the lowest 10th percentile of importance
threshold = np.percentile(mean_abs_shap_values, 50)
len(feature_importances)
# Identify features below the threshold
features_to_remove = [
    feature_names[i]
    for i, v in enumerate(mean_abs_shap_values)
    if v < np.percentile(mean_abs_shap_values, 50)
]
features_to_keep = [
    feature_names[i]
    for i, v in enumerate(mean_abs_shap_values)
    if v >= np.percentile(mean_abs_shap_values, 50)
]


top_features = [
    feature_names[i]
    for i, v in enumerate(mean_abs_shap_values)
    if v >= np.percentile(mean_abs_shap_values, 70)
]
worst_features = [
    feature_names[i]
    for i, v in enumerate(mean_abs_shap_values)
    if v < np.percentile(mean_abs_shap_values, 30)
]


len(top_features)
best_expanded = []
for tf in top_features:
    l = [x.split('_') for x in feature_names if tf in x][0]
    best_expanded.extend(l)
    
best_expanded = Counter(best_expanded)
best_expanded_df = pd.DataFrame(best_expanded,index=[0]).T.sort_values(0,ascending=False).rename(columns={0:'best'})

len(worst_features)
worst_expanded = []
for tf in worst_features:
    l = [x.split('_') for x in feature_names if tf in x][0]
    worst_expanded.extend(l)
    
worst_expanded = Counter(worst_expanded)
worst_expanded_df = pd.DataFrame(worst_expanded,index=[0]).T.sort_values(0,ascending=False).rename(columns={0:'worst'})



 


# outer merge
merged = pd.merge(best_expanded_df, worst_expanded_df, how='outer', left_index=True, right_index=True).fillna(0)
# remove any index that starts with 'e'
merged = merged[~merged.index.str.startswith('e')]
# remove 'P' and 'C'
merged = merged[~merged.index.isin(['P','C'])]

merged.sort_values(['best','worst'],ascending=[False,True]).plot(kind='bar',figsize=(12,10))



featuers_split = []
for f in feature_names:
    for x in f.split("_"):
        featuers_split.append(x)

set(featuers_split)


fx_importances = {}
for fx in set(featuers_split):
    fx_importances[fx] = 0
    for fi, fn in feature_importances:
        if fx in fn:
            fx_importances[fx] += fi


def plot_time_lag_features(plot_dict, scale=0.0002):
    # order dicttionary by key
    plot_dict = dict(sorted(plot_dict.items()))
    # order dicttionary by value 
    # plot_dict = dict(sorted(plot_dict.items(), key=lambda item: item[1], reverse=True))
    
    # barplot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(plot_dict.keys(), plot_dict.values())
    # rotate x labels
    plt.xticks(rotation=90)
    # max y vlue 3
    plt.ylim(0,scale)
    plt.show()


def importance_dict(sw="tn", l=3):
    features = {}
    for k in fx_importances:
        if k.startswith(sw) and len(k) == l:
            features.update({k: fx_importances[k]})
    return features


plot_time_lag_features(importance_dict("tn", 3),scale=3)
plot_time_lag_features(importance_dict("e", 2),scale=5)
plot_time_lag_features(importance_dict("b", 2))


imp = pd.DataFrame(fx_importances,index=[0]).T.sort_values(0,ascending=False)
exp_values = [x for x in imp['index'].values if x.startswith('e')]
imp[imp['index'].isin(exp_values)]
imp[~imp['index'].isin(exp_values)]

imp.plot(kind='bar',figsize=(12,10))
    
desc_stat = ['Sum', 'Mean', 'Med', 'Mode', 'Std', 'Var', 'Range', 'Q1', 'Q3', 'IQR', 'Skew', 'Kurt']
# filter index that is in desc_stat
imp_plot = imp[imp.index.isin(desc_stat)]
imp.plot(kind='bar',figsize=(12,10))




fx_importances = {}
for fx in set(featuers_split):
    fx_importances[fx] = 0
    for fi, fn in feature_importances:
        if fx in fn:
            fx_importances[fx] += fi


import re




# Regular expression pattern
pattern = r"tn\d+_e\d+_tV_b\d+_C"


patterns = [r"^tn\d+_e\d+_bS",r"^tn\d+_e\d+_aS",r"^tn\d+_e\d+_oI",r"^tn\d+_e\d+_tV"]


extra_pt = {}
for pattern, na in zip(patterns, ["bS", "aS", "oI", "tV"]):
    # Loop through the list and print matches
    expiration_tv_importance = {}
    pt = 0
    for fi, featue in feature_importances:
        if re.match(pattern, featue):
            fs = featue.rsplit('_',3)[0]
            # print(fs)
            if featue.rsplit('_',2)[0] not in expiration_tv_importance:
                expiration_tv_importance[fs] = 0
            expiration_tv_importance[fs] += fi
            pt += fi
            
    extra_pt[na] = pt


    plot_time_lag_features(expiration_tv_importance)
    
plot_time_lag_features(extra_pt, scale=5)
