package dev.ericson.neuroscape.api.dto;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;

import dev.ericson.neuroscape.data.DatasetType;
import dev.ericson.neuroscape.model.SavedModel;
import dev.ericson.neuroscape.nn.Activation;
import dev.ericson.neuroscape.nn.OptimizerType;

/** Metadata of a persisted model (weights are not exposed). */
public record ModelSummary(
        long id,
        String name,
        DatasetType dataset,
        List<Integer> hiddenLayers,
        Activation activation,
        OptimizerType optimizer,
        double learningRate,
        int epoch,
        double testLoss,
        double testAccuracy,
        Instant createdAt) {

    public static ModelSummary of(SavedModel m) {
        List<Integer> layers = m.getHiddenLayers().isBlank()
                ? List.of()
                : Arrays.stream(m.getHiddenLayers().split(",")).map(Integer::valueOf).toList();
        return new ModelSummary(m.getId(), m.getName(), m.getDataset(), layers, m.getActivation(), m.getOptimizer(),
                m.getLearningRate(), m.getEpoch(), m.getTestLoss(), m.getTestAccuracy(), m.getCreatedAt());
    }
}
