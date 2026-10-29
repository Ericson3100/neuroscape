package dev.ericson.neuroscape.training;

/** Loss and accuracy after a given epoch, on both splits. */
public record MetricPoint(int epoch, double trainLoss, double testLoss, double trainAccuracy, double testAccuracy) {
}
