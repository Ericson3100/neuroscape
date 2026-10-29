package dev.ericson.neuroscape.model;

import java.time.Clock;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import dev.ericson.neuroscape.nn.Network;
import dev.ericson.neuroscape.nn.NetworkCodec;
import dev.ericson.neuroscape.training.MetricPoint;
import dev.ericson.neuroscape.training.Prediction;
import dev.ericson.neuroscape.training.TrainingConfig;
import dev.ericson.neuroscape.training.TrainingService;
import dev.ericson.neuroscape.training.TrainingSession;

/** Persists trained networks and turns them back into live sessions. */
@Service
@Transactional(readOnly = true)
public class ModelService {

    private final SavedModelRepository repository;
    private final TrainingService trainingService;
    private final Clock clock;

    @Autowired
    public ModelService(SavedModelRepository repository, TrainingService trainingService) {
        this(repository, trainingService, Clock.systemUTC());
    }

    ModelService(SavedModelRepository repository, TrainingService trainingService, Clock clock) {
        this.repository = repository;
        this.trainingService = trainingService;
        this.clock = clock;
    }

    @Transactional
    public SavedModel save(String sessionId, String name) {
        TrainingSession session = trainingService.get(sessionId);
        TrainingConfig config = session.config();
        MetricPoint metrics = session.latestMetrics();
        Network network = session.networkCopy();
        String layers = config.hiddenLayers().stream().map(String::valueOf).collect(Collectors.joining(","));
        var model = new SavedModel(name.strip(), config.dataset(), config.samples(), config.noise(), config.seed(),
                layers, config.activation(), config.optimizer(), config.learningRate(), config.batchSize(),
                metrics.epoch(), metrics.testLoss(), metrics.testAccuracy(), clock.instant(),
                NetworkCodec.encode(network));
        return repository.save(model);
    }

    public List<SavedModel> list() {
        return repository.findAllByOrderByCreatedAtDesc();
    }

    public SavedModel get(long id) {
        return repository.findById(id).orElseThrow(() -> new ModelNotFoundException(id));
    }

    @Transactional
    public void delete(long id) {
        if (!repository.existsById(id)) {
            throw new ModelNotFoundException(id);
        }
        repository.deleteById(id);
    }

    /** Starts a new live session seeded with the saved weights, so training can continue from there. */
    public TrainingSession load(long id) {
        SavedModel model = get(id);
        return trainingService.resume(configOf(model), NetworkCodec.decode(model.getParameters()), model.getEpoch());
    }

    public Prediction predict(long id, double x, double y) {
        return Prediction.of(NetworkCodec.decode(get(id).getParameters()), x, y);
    }

    static TrainingConfig configOf(SavedModel model) {
        List<Integer> layers = model.getHiddenLayers().isBlank()
                ? List.of()
                : Arrays.stream(model.getHiddenLayers().split(",")).map(Integer::valueOf).toList();
        return new TrainingConfig(model.getDataset(), model.getSamples(), model.getNoise(), layers,
                model.getActivation(), model.getOptimizer(), model.getLearningRate(), model.getBatchSize(),
                model.getSeed());
    }
}
