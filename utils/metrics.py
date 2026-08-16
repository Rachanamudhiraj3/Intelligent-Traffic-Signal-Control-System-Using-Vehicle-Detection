import numpy as np
import pandas as pd


class Metrics:
    def __init__(self):
        pass

    # Demo System Metrics
    def calculate_metrics(self, actual, predicted):

        tp = fp = tn = fn = 0

        for a, p in zip(actual, predicted):

            if a and p:
                tp += 1

            elif not a and not p:
                tn += 1

            elif not a and p:
                fp += 1

            elif a and not p:
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        confusion = np.array([
            [tn, fp],
            [fn, tp]
        ])

        return (
            accuracy,
            precision,
            recall,
            f1,
            confusion
        )

    # Read YOLO Training Results
    def print_model_metrics(self, csv_path):

        try:

            df = pd.read_csv(csv_path)

            last = df.iloc[-1]

            precision = last["metrics/precision(B)"] * 100
            recall = last["metrics/recall(B)"] * 100
            map50 = last["metrics/mAP50(B)"] * 100
            map5095 = last["metrics/mAP50-95(B)"] * 100

            print("\n============== MODEL EVALUATION ==============\n")

            print(f"Overall Model Accuracy (mAP@0.50) : {map50:.2f}%\n")

            print(f"Precision      : {precision:.2f}%")
            print(f"Recall         : {recall:.2f}%")
            print(f"mAP@0.50       : {map50:.2f}%")
            print(f"mAP@0.50:0.95  : {map5095:.2f}%")

            print("\n==============================================")

        except Exception as e:

            print("\nUnable to load YOLO results.csv")
            print(e)