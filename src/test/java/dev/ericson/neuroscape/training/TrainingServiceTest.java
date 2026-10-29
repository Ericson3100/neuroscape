package dev.ericson.neuroscape.training;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import dev.ericson.neuroscape.config.NeuroscapeProperties;
import dev.ericson.neuroscape.data.DatasetType;
import dev.ericson.neuroscape.nn.Activation;
import dev.ericson.neuroscape.nn.OptimizerType;

class TrainingServiceTest {

    private final MutableClock clock = new MutableClock();
    private TrainingService service;

    @BeforeEach
    void setUp() {
        var properties = new NeuroscapeProperties(16, 1000, 3, 2, Duration.ofMinutes(5), Duration.ofMinutes(1), 10);
        service = new TrainingService(properties, clock);
    }

    private static TrainingConfig config() {
        return new TrainingConfig(DatasetType.CIRCLES, 200, 0.05, List.of(6, 6), Activation.TANH,
                OptimizerType.ADAM, 0.03, 16, 1L);
    }

    @Test
    void tickTrainsOnlyRunningSessions() {
        TrainingSession running = service.create(config());
        TrainingSession paused = service.create(config());
        service.play(running.id());

        service.tick();

        assertThat(running.epoch()).isEqualTo(3);
        assertThat(paused.epoch()).isZero();
    }

    @Test
    void snapshotCoversTheGrid() {
        TrainingSession session = service.create(config());
        TrainingSnapshot snapshot = service.step(session.id(), 5);

        assertThat(snapshot.grid()).hasSize(16 * 16);
        assertThat(snapshot.layers()).hasSize(3);
        assertThat(snapshot.metrics().epoch()).isEqualTo(5);
        for (float p : snapshot.grid()) {
            assertThat(p).isBetween(0f, 1f);
        }
    }

    @Test
    void resetRestoresInitialWeightsDeterministically() {
        TrainingSession session = service.create(config());
        double initialLoss = session.latestMetrics().trainLoss();
        service.step(session.id(), 20);
        assertThat(session.latestMetrics().trainLoss()).isLessThan(initialLoss);

        service.reset(session.id());

        assertThat(session.epoch()).isZero();
        assertThat(session.latestMetrics().trainLoss()).isEqualTo(initialLoss);
        assertThat(session.history()).hasSize(1);
    }

    @Test
    void historyIsBounded() {
        TrainingSession session = service.create(config());
        for (int i = 0; i < 50; i++) {
            service.step(session.id(), 1);
        }
        assertThat(session.history().size()).isLessThanOrEqualTo(11);
        assertThat(session.history().getLast().epoch()).isEqualTo(50);
    }

    @Test
    void enforcesSessionLimitAndEvictsIdleSessions() {
        service.create(config());
        service.create(config());
        assertThatThrownBy(() -> service.create(config())).isInstanceOf(TooManySessionsException.class);

        clock.advance(Duration.ofMinutes(10));
        assertThat(service.create(config())).isNotNull();
        assertThat(service.sessions()).hasSize(1);
    }

    @Test
    void unknownSessionIsReported() {
        assertThatThrownBy(() -> service.get("nope")).isInstanceOf(SessionNotFoundException.class);
    }

    private static final class MutableClock extends Clock {
        private Instant now = Instant.parse("2026-01-01T00:00:00Z");

        void advance(Duration d) {
            now = now.plus(d);
        }

        @Override
        public java.time.ZoneId getZone() {
            return ZoneOffset.UTC;
        }

        @Override
        public Clock withZone(java.time.ZoneId zone) {
            return this;
        }

        @Override
        public Instant instant() {
            return now;
        }
    }
}
