import pandas as pd
import numpy as np
from models.sign_model import SignModel


def _dtw_distance(sequence_a, sequence_b):
    """Exact DTW using linear memory and Euclidean frame-to-frame distance."""
    if len(sequence_a) == 0 or len(sequence_b) == 0:
        return np.inf

    previous = np.full(len(sequence_b) + 1, np.inf)
    previous[0] = 0.0
    for frame_a in sequence_a:
        current = np.full(len(sequence_b) + 1, np.inf)
        for j, frame_b in enumerate(sequence_b, start=1):
            cost = np.linalg.norm(np.asarray(frame_a) - np.asarray(frame_b))
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(previous[-1])


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    for idx, row in reference_signs.iterrows():
        ref_sign_model = row["sign_model"]

        # Compare only references that use the same hand or hands.
        distance = 0.0
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ) and (recorded_sign.has_left_hand or recorded_sign.has_right_hand):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            if recorded_sign.has_left_hand:
                distance += _dtw_distance(rec_left_hand, ref_left_hand)
            if recorded_sign.has_right_hand:
                distance += _dtw_distance(rec_right_hand, ref_right_hand)

        # If not, distance equals infinity
        else:
            distance = np.inf
        reference_signs.at[idx, "distance"] = distance
    return reference_signs.sort_values(by=["distance"])
