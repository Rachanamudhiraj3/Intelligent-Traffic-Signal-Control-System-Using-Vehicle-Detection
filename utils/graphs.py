import matplotlib.pyplot as plt
class Graphs:
    def plot_model_evaluation(self):
        metrics = [
            "Precision",
            "Recall",
            "mAP@50",
            "mAP@50-95"
        ]
        values = [
            96.93,
            90.51,
            97.52,
            77.22
        ]
        plt.figure(figsize=(6,5))

        plt.bar(
            metrics,
            values
        )
        plt.ylim(0,100)
        plt.ylabel("Score (%)")
        plt.title(
            "YOLOv8 Emergency Vehicle Detection Model Evaluation"
        )

        for i, value in enumerate(values):
            plt.text(
                i,
                value + 1,
                f"{value:.2f}%",
                ha="center",
                fontsize=9
            )

        plt.tight_layout()

        plt.savefig(
            "output/model_evaluation.png",
            dpi=300
        )
        plt.show()

    def plot_comparison(self):

        metrics = [
            "Accuracy",
            "mAP50",
            "Precision",
            "Recall",
            "F1-Score"
        ]

        existing = [
            95,
            96,
            93,
            92,
            93
        ]

        proposed = [
            97.52,
            97.52,
            96.93,
            90.51,
            93.60
        ]

        plt.figure(figsize=(7,5))

        plt.plot(
            metrics,
            existing,
            marker="o",
            linewidth=2,
            label="Existing Method"
        )

        plt.plot(
            metrics,
            proposed,
            marker="o",
            linewidth=2,
            label="Proposed Method"
        )

        plt.ylim(80,100)

        plt.ylabel("Score (%)")

        plt.title(
            "Performance Comparison: Existing vs Proposed Model"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "output/performance_comparison.png",
            dpi=300
        )

        plt.show()