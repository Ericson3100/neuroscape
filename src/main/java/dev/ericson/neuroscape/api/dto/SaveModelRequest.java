package dev.ericson.neuroscape.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/** Snapshot the current weights of a live session under a human-readable name. */
public record SaveModelRequest(@NotBlank String sessionId, @NotBlank @Size(max = 80) String name) {
}
