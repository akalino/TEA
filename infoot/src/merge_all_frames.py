import os
import pandas as pd


def find_results():
    files = os.listdir()
    csvs = [x for x in files if 'results' in x]
    all_frames = []
    for _f in csvs:
        _df = pd.read_csv(_f)
        _df.columns = ['source_model', 'target_model',
                       'batch_size', 'h10', 'was_dist']
        all_frames.append(_df)
    out = pd.concat(all_frames)
    out.sort_values(by=['source_model', 'target_model', 'batch_size'],
                    inplace=True)
    return out


if __name__ == "__main__":
    df = find_results()
    print(df.head(100))
