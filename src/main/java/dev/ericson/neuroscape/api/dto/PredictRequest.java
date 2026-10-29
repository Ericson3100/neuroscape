package dev.ericson.neuroscape.api.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;

/** A point in the input plane to classify. */
public record PredictRequest(
        @NotNull @DecimalMin("-10.0") @DecimalMax("10.0") Double x,
        @NotNull @DecimalMin("-10.0") @DecimalMax("10.0") Double y) {
}
