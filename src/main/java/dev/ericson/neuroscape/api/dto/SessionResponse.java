package dev.ericson.neuroscape.api.dto;

import java.util.List;

import dev.ericson.neuroscape.data.Sample;
import dev.ericson.neuroscape.training.MetricPoint;
import dev.ericson.neuroscape.training.TrainingConfig;
import dev.ericson.neuroscape.training.TrainingSession;
import dev.ericson.neuroscape.training.TrainingSnapshot;

/** Full state of a session: config, dataset, metric history and the current frame. */
public record SessionResponse(
        String id,
        TrainingConfig config,
        List<Sample> train,
        List<Sample> test,
        List<MetricPoint> history,
        TrainingSnapshot snapshot) {

    public static SessionResponse of(TrainingSession session, TrainingSnapshot snapshot) {
        return new SessionResponse(session.id(), session.config(), session.dataset().train(),
                session.dataset().test(), session.history(), snapshot);
    }
}
