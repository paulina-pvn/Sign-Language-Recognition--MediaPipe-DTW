import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        # pandas 3 may expose the column's NumPy values as read-only.
        # Assign a new column instead of mutating that array view.
        self.reference_signs["distance"] = 0.0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=1, threshold=0.2):
        """
        Vote among the closest reference videos and return a sign only when its
        share of the vote reaches the threshold.

        :param batch_size: Number of closest reference videos to include in the vote.
        :param threshold: Minimum share of votes required to return a sign label.
        :return: The name of the predicted sign
        """
        if batch_size <= 0 or self.reference_signs.empty:
            return "Signe inconnu"

        # There may be fewer reference videos than the requested batch size.
        effective_batch_size = min(batch_size, len(self.reference_signs))

        # Get the names attached to the closest reference videos.
        sign_names = self.reference_signs.iloc[:effective_batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / effective_batch_size < threshold:
            return "Signe inconnu"
        return predicted_sign
