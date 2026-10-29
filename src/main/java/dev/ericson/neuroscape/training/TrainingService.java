package dev.ericson.neuroscape.training;

import java.io.IOException;
import java.time.Clock;
import java.time.Instant;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import dev.ericson.neuroscape.config.NeuroscapeProperties;
import dev.ericson.neuroscape.nn.Network;

/**
 * Owns all live sessions and the background training loop. Every tick, each running session trains
 * a few epochs and its snapshot is pushed to all SSE subscribers.
 */
@Service
public class TrainingService {

    private static final Logger log = LoggerFactory.getLogger(TrainingService.class);
    static final String SNAPSHOT_EVENT = "snapshot";

    private final NeuroscapeProperties properties;
    private final Clock clock;
    private final Map<String, TrainingSession> sessions = new ConcurrentHashMap<>();
    private ScheduledExecutorService scheduler;

    @Autowired
    public TrainingService(NeuroscapeProperties properties) {
        this(properties, Clock.systemUTC());
    }

    TrainingService(NeuroscapeProperties properties, Clock clock) {
        this.properties = properties;
        this.clock = clock;
    }

    @PostConstruct
    void start() {
        scheduler = Executors.newSingleThreadScheduledExecutor(Thread.ofPlatform().name("nn-trainer").daemon().factory());
        scheduler.scheduleAtFixedRate(this::tickSafely, properties.tickMillis(), properties.tickMillis(), TimeUnit.MILLISECONDS);
        scheduler.scheduleAtFixedRate(this::evictIdle, 1, 1, TimeUnit.MINUTES);
    }

    @PreDestroy
    void stop() {
        scheduler.shutdownNow();
        sessions.values().forEach(s -> s.subscribers().forEach(SseEmitter::complete));
    }

    public TrainingSession create(TrainingConfig config) {
        Network network = Network.create(2, config.hiddenLayers(), config.activation(), new Random(config.seed()));
        return register(config, network, 0);
    }

    /** Resumes a previously trained network (e.g. loaded from the database). */
    public TrainingSession resume(TrainingConfig config, Network network, int epoch) {
        return register(config, network, epoch);
    }

    private TrainingSession register(TrainingConfig config, Network network, int epoch) {
        if (sessions.size() >= properties.maxSessions()) {
            evictIdle();
            if (sessions.size() >= properties.maxSessions()) {
                throw new TooManySessionsException(properties.maxSessions());
            }
        }
        String id = UUID.randomUUID().toString();
        var session = new TrainingSession(id, config, network, epoch, properties.maxHistory(), clock);
        sessions.put(id, session);
        return session;
    }

    public TrainingSession get(String id) {
        TrainingSession session = sessions.get(id);
        if (session == null) {
            throw new SessionNotFoundException(id);
        }
        session.touch();
        return session;
    }

    public void delete(String id) {
        TrainingSession session = sessions.remove(id);
        if (session == null) {
            throw new SessionNotFoundException(id);
        }
        session.subscribers().forEach(SseEmitter::complete);
    }

    public TrainingSnapshot play(String id) {
        TrainingSession session = get(id);
        session.setRunning(true);
        return publish(session);
    }

    public TrainingSnapshot pause(String id) {
        TrainingSession session = get(id);
        session.setRunning(false);
        return publish(session);
    }

    public TrainingSnapshot step(String id, int epochs) {
        TrainingSession session = get(id);
        session.train(epochs);
        return publish(session);
    }

    public TrainingSnapshot reset(String id) {
        TrainingSession session = get(id);
        session.reset();
        return publish(session);
    }

    public TrainingSnapshot snapshot(String id) {
        return get(id).snapshot(properties.gridResolution());
    }

    public Prediction predict(String id, double x, double y) {
        return get(id).predict(x, y);
    }

    public SseEmitter subscribe(String id) {
        TrainingSession session = get(id);
        SseEmitter emitter = new SseEmitter(properties.sseTimeout().toMillis());
        List<SseEmitter> subscribers = session.subscribers();
        subscribers.add(emitter);
        emitter.onCompletion(() -> subscribers.remove(emitter));
        emitter.onTimeout(() -> subscribers.remove(emitter));
        emitter.onError(e -> subscribers.remove(emitter));
        send(emitter, session.snapshot(properties.gridResolution()), subscribers);
        return emitter;
    }

    public Collection<TrainingSession> sessions() {
        return List.copyOf(sessions.values());
    }

    private void tickSafely() {
        try {
            tick();
        } catch (RuntimeException e) {
            log.error("Training tick failed", e);
        }
    }

    void tick() {
        for (TrainingSession session : sessions.values()) {
            if (session.isRunning()) {
                session.train(properties.epochsPerTick());
                publish(session);
            }
        }
    }

    private TrainingSnapshot publish(TrainingSession session) {
        TrainingSnapshot snapshot = session.snapshot(properties.gridResolution());
        for (SseEmitter emitter : session.subscribers()) {
            send(emitter, snapshot, session.subscribers());
        }
        return snapshot;
    }

    private static void send(SseEmitter emitter, TrainingSnapshot snapshot, List<SseEmitter> subscribers) {
        try {
            emitter.send(SseEmitter.event().name(SNAPSHOT_EVENT).data(snapshot, MediaType.APPLICATION_JSON));
        } catch (IOException | IllegalStateException e) {
            // Client went away; drop it silently, the browser will reconnect if it still cares.
            subscribers.remove(emitter);
        }
    }

    void evictIdle() {
        Instant cutoff = clock.instant().minus(properties.sessionIdleTimeout());
        sessions.values().removeIf(s -> s.subscribers().isEmpty() && s.lastActivity().isBefore(cutoff));
    }
}
