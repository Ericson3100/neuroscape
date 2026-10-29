package dev.ericson.neuroscape.api;

import java.net.URI;
import java.util.concurrent.ThreadLocalRandom;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import dev.ericson.neuroscape.api.dto.CreateSessionRequest;
import dev.ericson.neuroscape.api.dto.PredictRequest;
import dev.ericson.neuroscape.api.dto.SessionResponse;
import dev.ericson.neuroscape.training.Prediction;
import dev.ericson.neuroscape.training.TrainingService;
import dev.ericson.neuroscape.training.TrainingSession;
import dev.ericson.neuroscape.training.TrainingSnapshot;

/** Live training sessions: create, control, stream and query. */
@RestController
@RequestMapping("/api/sessions")
public class SessionController {

    private final TrainingService trainingService;

    public SessionController(TrainingService trainingService) {
        this.trainingService = trainingService;
    }

    @PostMapping
    public ResponseEntity<SessionResponse> create(@Valid @RequestBody CreateSessionRequest request) {
        TrainingSession session = trainingService.create(request.toConfig(ThreadLocalRandom.current().nextLong()));
        SessionResponse body = SessionResponse.of(session, trainingService.snapshot(session.id()));
        return ResponseEntity.created(URI.create("/api/sessions/" + session.id())).body(body);
    }

    @GetMapping("/{id}")
    public SessionResponse get(@PathVariable String id) {
        TrainingSession session = trainingService.get(id);
        return SessionResponse.of(session, trainingService.snapshot(id));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable String id) {
        trainingService.delete(id);
        return ResponseEntity.noContent().build();
    }

    @GetMapping(path = "/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@PathVariable String id) {
        return trainingService.subscribe(id);
    }

    @PostMapping("/{id}/play")
    public TrainingSnapshot play(@PathVariable String id) {
        return trainingService.play(id);
    }

    @PostMapping("/{id}/pause")
    public TrainingSnapshot pause(@PathVariable String id) {
        return trainingService.pause(id);
    }

    @PostMapping("/{id}/step")
    public TrainingSnapshot step(@PathVariable String id,
            @RequestParam(defaultValue = "1") @Min(1) @Max(1000) int epochs) {
        return trainingService.step(id, epochs);
    }

    @PostMapping("/{id}/reset")
    public TrainingSnapshot reset(@PathVariable String id) {
        return trainingService.reset(id);
    }

    @PostMapping("/{id}/predict")
    public Prediction predict(@PathVariable String id, @Valid @RequestBody PredictRequest request) {
        return trainingService.predict(id, request.x(), request.y());
    }
}
