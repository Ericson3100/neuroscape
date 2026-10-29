package dev.ericson.neuroscape.training;

import java.util.List;

import dev.ericson.neuroscape.data.DatasetType;
import dev.ericson.neuroscape.nn.Activation;
import dev.ericson.neuroscape.nn.OptimizerType;

/** Everything needed to (re)build a training session deterministically. */
public record TrainingConfig(
        DatasetType dataset,
        int samples,
        double noise,
        List<Integer> hiddenLayers,
        Activation activation,
        OptimizerType optimizer,
        double learningRate,
        int batchSize,
        long seed) {

    public TrainingConfig {
        hiddenLayers = List.copyOf(hiddenLayers);
    }
}
