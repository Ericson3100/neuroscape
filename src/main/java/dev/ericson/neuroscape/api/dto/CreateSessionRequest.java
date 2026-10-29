package dev.ericson.neuroscape.api.dto;

import java.util.List;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import dev.ericson.neuroscape.data.DatasetType;
import dev.ericson.neuroscape.nn.Activation;
import dev.ericson.neuroscape.nn.OptimizerType;
import dev.ericson.neuroscape.training.TrainingConfig;

/** Request body for {@code POST /api/sessions}. {@code seed} is optional; a random one is chosen if absent. */
public record CreateSessionRequest(
        @NotNull DatasetType dataset,
        @NotNull @Min(20) @Max(2000) Integer samples,
        @NotNull @DecimalMin("0.0") @DecimalMax("0.5") Double noise,
        @NotNull @Size(max = 6) List<@NotNull @Min(1) @Max(16) Integer> hiddenLayers,
        @NotNull Activation activation,
        @NotNull OptimizerType optimizer,
        @NotNull @DecimalMin(value = "0.0", inclusive = false) @DecimalMax("5.0") Double learningRate,
        @NotNull @Min(1) @Max(512) Integer batchSize,
        Long seed) {

    public TrainingConfig toConfig(long fallbackSeed) {
        return new TrainingConfig(dataset, samples, noise, hiddenLayers, activation, optimizer, learningRate,
                batchSize, seed != null ? seed : fallbackSeed);
    }
}
