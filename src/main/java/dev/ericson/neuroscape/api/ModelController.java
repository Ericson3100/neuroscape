package dev.ericson.neuroscape.api;

import java.net.URI;
import java.util.List;

import jakarta.validation.Valid;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import dev.ericson.neuroscape.api.dto.ModelSummary;
import dev.ericson.neuroscape.api.dto.PredictRequest;
import dev.ericson.neuroscape.api.dto.SaveModelRequest;
import dev.ericson.neuroscape.api.dto.SessionResponse;
import dev.ericson.neuroscape.model.ModelService;
import dev.ericson.neuroscape.model.SavedModel;
import dev.ericson.neuroscape.training.Prediction;
import dev.ericson.neuroscape.training.TrainingService;
import dev.ericson.neuroscape.training.TrainingSession;

/** Saved (persisted) models. */
@RestController
@RequestMapping("/api/models")
public class ModelController {

    private final ModelService modelService;
    private final TrainingService trainingService;

    public ModelController(ModelService modelService, TrainingService trainingService) {
        this.modelService = modelService;
        this.trainingService = trainingService;
    }

    @GetMapping
    public List<ModelSummary> list() {
        return modelService.list().stream().map(ModelSummary::of).toList();
    }

    @PostMapping
    public ResponseEntity<ModelSummary> save(@Valid @RequestBody SaveModelRequest request) {
        SavedModel model = modelService.save(request.sessionId(), request.name());
        return ResponseEntity.created(URI.create("/api/models/" + model.getId())).body(ModelSummary.of(model));
    }

    @GetMapping("/{id}")
    public ModelSummary get(@PathVariable long id) {
        return ModelSummary.of(modelService.get(id));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable long id) {
        modelService.delete(id);
        return ResponseEntity.noContent().build();
    }

    /** Opens a new live session initialised with the saved weights. */
    @PostMapping("/{id}/load")
    public ResponseEntity<SessionResponse> load(@PathVariable long id) {
        TrainingSession session = modelService.load(id);
        SessionResponse body = SessionResponse.of(session, trainingService.snapshot(session.id()));
        return ResponseEntity.created(URI.create("/api/sessions/" + session.id())).body(body);
    }

    @PostMapping("/{id}/predict")
    public Prediction predict(@PathVariable long id, @Valid @RequestBody PredictRequest request) {
        return modelService.predict(id, request.x(), request.y());
    }
}
