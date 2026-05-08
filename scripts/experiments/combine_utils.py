import pandas as pd


def combine_result_files(interim_res_path, final_csv_path, result_files):
    """Load, align, concatenate, and save a set of result CSV files."""
    frames = []
    reference_columns = None

    for result_file in result_files:
        frame = pd.read_csv(interim_res_path / result_file, index_col=0)
        if reference_columns is None:
            reference_columns = frame.columns
        else:
            frame = frame[reference_columns]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=False)
    combined.to_csv(final_csv_path)
    return combined