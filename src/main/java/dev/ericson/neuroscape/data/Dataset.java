package dev.ericson.neuroscape.data;

import java.util.List;

/** A dataset split into training and held-out test samples, with cached feature/label arrays. */
public record Dataset(DatasetType type, List<Sample> train, List<Sample> test) {

    public Dataset {
        train = List.copyOf(train);
        test = List.copyOf(test);
    }

    public double[][] trainFeatures() {
        return features(train);
    }

    public int[] trainLabels() {
        return labels(train);
    }

    public double[][] testFeatures() {
        return features(test);
    }

    public int[] testLabels() {
        return labels(test);
    }

    private static double[][] features(List<Sample> samples) {
        return samples.stream().map(Sample::features).toArray(double[][]::new);
    }

    private static int[] labels(List<Sample> samples) {
        return samples.stream().mapToInt(Sample::label).toArray();
    }
}
