import argparse
import pandas as pd
from supernnova.utils import logging_utils as lu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Aggregating stats for CNNs")

    parser.add_argument(
        "--dump_dir",
        type=str,
        default=f"../cnndump/",
        help="Default path where CNN dump is",
    )
    args = parser.parse_args()

    # read summary stats
    df_stats = pd.read_csv(f"{args.dump_dir}/stats/summary_stats.csv")

    # get hp stats
    df_hp = df_stats[(df_stats['source_data'] == 'saltfit') & (
        df_stats['model_name_noseed'].str.contains('DF_0.2')) & (df_stats['model_name_noseed'].str.contains('CNN'))]
    # highest accuracy complete lc
    maxacc = df_hp['all_accuracy_mean'].max()
    lu.print_green(f'HP highest accuracy all {maxacc}')
    print(df_hp[df_hp['all_accuracy_mean'] == maxacc]
          ['model_name_noseed'].values)

    # best configuration
    df_best = df_stats[(df_stats['model_name_noseed'].str.contains('CNN')) & (df_stats['model_name_noseed'].str.contains('DF_1.0'))]
    # df_best = df_best.round(2)
    pd.set_option('max_colwidth', 400)

    # all accuracy
    lu.print_green('best accuracy all saltfit')
    df_tmp = df_best[(
        df_stats['model_name_noseed'].str.contains('saltfit'))].copy()
    print(df_tmp[['model_name_noseed', 'all_accuracy_mean',
                  'all_accuracy_std', 'all_auc_mean', 'all_auc_std']])
    lu.print_green('best accuracy all photometry')
    df_tmp = df_best[(
        df_stats['model_name_noseed'].str.contains('photometry'))].copy()
    print(df_tmp[['model_name_noseed', 'all_accuracy_mean', 'all_accuracy_std']])

    # accuracy around peak
    for peak in ["-2", "0", "+2"]:
        lu.print_green(f'peak{peak}')
        print(df_best[['model_name_noseed',
                       f'{peak}_accuracy_mean', f'{peak}_accuracy_std']])

    # import ipdb; ipdb.set_trace()
