import numpy as np
import pandas as pd

def assign_mean(col, fixed_pcd_df, moving_pcd_df, matching_pairs, fill_val=0):
    result_df = pd.DataFrame(index=fixed_pcd_df.index)
    for idx in range(len(fixed_pcd_df)):
        matched_indices = matching_pairs[matching_pairs[:, 0] == idx][:, 1]
        if not len(matched_indices):
                aggregated_values = [fill_val]
        else:
            aggregated_values = [moving_pcd_df.loc[i, col] for i in matched_indices if i in moving_pcd_df.index]
        
        result_df.loc[idx, col] = np.mean(aggregated_values)
    return result_df

def _matching_eval_mask(
    matched_track_1: np.ndarray,
    matched_track_2: np.ndarray,
    ignore_label=-1,
) -> np.ndarray:
    return (matched_track_1 != ignore_label) | (matched_track_2 != ignore_label)


def accuracy(
    reg_gt_1,
    reg_gt_2,
    row_ind,
    col_ind,
    ignore_label=-1,
):
    matched_label_1 = reg_gt_1[row_ind]
    matched_label_2 = reg_gt_2[col_ind]

    pairs_match = matched_label_1 == matched_label_2
    eval_mask = _matching_eval_mask(
        matched_label_1,
        matched_label_2,
        ignore_label=ignore_label,
    )
    accuracy = np.sum(pairs_match[eval_mask]) / len(pairs_match[eval_mask])

    return accuracy
