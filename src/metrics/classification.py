"""Classification metrics (precision, recall, F1) shared by the experiments.

Predictions that the model did not express in the expected format are mapped to the
``UNPARSED`` pseudo class, so that they always count as errors and never inflate the
precision of a real class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNPARSED = "UNPARSED"


def _as_label(value: object) -> str:
    if value is None:
        return UNPARSED
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


@dataclass
class ClassMetrics:
    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def support(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass
class ClassificationReport:
    success: int = 0
    failed: int = 0
    unparsed: int = 0
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.success + self.failed

    @property
    def accuracy(self) -> float:
        return self.success / self.total if self.total else 0.0

    @property
    def macro_precision(self) -> float:
        return self._mean("precision")

    @property
    def macro_recall(self) -> float:
        return self._mean("recall")

    @property
    def macro_f1(self) -> float:
        return self._mean("f1")

    @property
    def weighted_f1(self) -> float:
        classes = self._expected_classes()
        support = sum(metrics.support for metrics in classes)
        if not support:
            return 0.0
        return sum(metrics.f1 * metrics.support for metrics in classes) / support

    def _expected_classes(self) -> list[ClassMetrics]:
        """Only the classes that appear in the ground truth are averaged."""
        return [metrics for metrics in self.per_class.values() if metrics.support > 0]

    def _mean(self, attribute: str) -> float:
        classes = self._expected_classes()
        if not classes:
            return 0.0
        return sum(getattr(metrics, attribute) for metrics in classes) / len(classes)

    def format(self, title: str = "Results") -> str:
        lines = [
            f"{title}: failed={self.failed}, success={self.success}, unparsed={self.unparsed}",
            f"Accuracy: {self.accuracy:.2f}",
            f"Macro precision: {self.macro_precision:.2f}, "
            f"macro recall: {self.macro_recall:.2f}, macro F1: {self.macro_f1:.2f}",
            f"Weighted F1: {self.weighted_f1:.2f}",
            f"{'class':<12}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}",
        ]
        for label, metrics in sorted(self.per_class.items()):
            lines.append(
                f"{label:<12}{metrics.precision:>10.2f}{metrics.recall:>8.2f}"
                f"{metrics.f1:>8.2f}{metrics.support:>9}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def compute_report(pairs: list[tuple[object, object]]) -> ClassificationReport:
    """Build a report from ``(expected, predicted)`` pairs."""
    report = ClassificationReport()
    for raw_expected, raw_predicted in pairs:
        expected = _as_label(raw_expected)
        predicted = _as_label(raw_predicted)
        for label in (expected, predicted):
            report.per_class.setdefault(label, ClassMetrics(label=label))
        if predicted == UNPARSED:
            report.unparsed += 1
        if expected == predicted:
            report.success += 1
            report.per_class[expected].true_positives += 1
        else:
            report.failed += 1
            report.per_class[predicted].false_positives += 1
            report.per_class[expected].false_negatives += 1
    return report
