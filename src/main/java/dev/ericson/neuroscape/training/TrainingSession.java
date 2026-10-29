package dev.ericson.neuroscape.training;

import java.time.Clock;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import dev.ericson.neuroscape.data.Dataset;
import dev.ericson.neuroscape.data.DatasetGenerator;
import dev.ericson.neuroscape.nn.Network;
import dev.ericson.neuroscape.nn.Optimizer;

/**
 * One live experiment: a dataset, a network, its optimizer state and metric history.
 * All mutating methods are synchronized — the scheduler thread trains while request threads read.
 */
public final class TrainingSession {

    /** Half-width of the square input region sampled for the decision surface. */
    public static final double EXTENT = 1.2;

    private final String id;
    private final TrainingConfig config;
    private final Dataset dataset;
    private final double[][] trainX;
    private final int[] trainY;
    private final double[][] testX;
    private final int[] testY;
    private final int maxHistory;
    private final Clock clock;
    private final List<SseEmitter> subscribers = new CopyOnWriteArrayList<>();

    private Network network;
    private Network initialNetwork;
    private int initialEpoch;
    private Optimizer optimizer;
    private Random shuffle;
    private int epoch;
    private boolean running;
    private List<MetricPoint> history = new ArrayList<>();
    private volatile Instant lastActivity;

    TrainingSession(String id, TrainingConfig config, Network network, int startEpoch, int maxHistory, Clock clock) {
        this.id = id;
        this.config = config;
        this.dataset = DatasetGenerator.generate(config.dataset(), config.samples(), config.noise(), config.seed());
        this.trainX = dataset.trainFeatures();
        this.trainY = dataset.trainLabels();
        this.testX = dataset.testFeatures();
        this.testY = dataset.testLabels();
        this.maxHistory = maxHistory;
        this.clock = clock;
        this.initialNetwork = network.copy();
        this.initialEpoch = startEpoch;
        restart();
    }

    private void restart() {
        network = initialNetwork.copy();
        optimizer = Optimizer.of(config.optimizer(), config.learningRate());
        shuffle = new Random(config.seed() ^ 0x5DEECE66DL);
        epoch = initialEpoch;
        running = false;
        history = new ArrayList<>();
        history.add(measure());
        touch();
    }

    public synchronized void train(int epochs) {
        for (int i = 0; i < epochs; i++) {
            network.trainEpoch(trainX, trainY, config.batchSize(), optimizer, shuffle);
            epoch++;
        }
        record(measure());
    }

    public synchronized void reset() {
        restart();
    }

    public synchronized void setRunning(boolean running) {
        this.running = running;
        touch();
    }

    public synchronized boolean isRunning() {
        return running;
    }

    public synchronized MetricPoint latestMetrics() {
        return history.getLast();
    }

    public synchronized List<MetricPoint> history() {
        return List.copyOf(history);
    }

    public synchronized int epoch() {
        return epoch;
    }

    /** Copy of the current network, safe to use outside the lock (e.g. for persisting). */
    public synchronized Network networkCopy() {
        return network.copy();
    }

    public synchronized Prediction predict(double x, double y) {
        touch();
        return Prediction.of(network, x, y);
    }

    public synchronized TrainingSnapshot snapshot(int resolution) {
        float[] grid = new float[resolution * resolution];
        double step = 2 * EXTENT / (resolution - 1);
        double[] input = new double[2];
        for (int row = 0; row < resolution; row++) {
            input[1] = -EXTENT + row * step;
            for (int col = 0; col < resolution; col++) {
                input[0] = -EXTENT + col * step;
                grid[row * resolution + col] = (float) network.predict(input);
            }
        }
        double[][][] w = network.weightsCopy();
        double[][] b = network.biasesCopy();
        List<TrainingSnapshot.LayerParameters> layers = new ArrayList<>(w.length);
        for (int l = 0; l < w.length; l++) {
            layers.add(new TrainingSnapshot.LayerParameters(w[l], b[l]));
        }
        return new TrainingSnapshot(id, running, history.getLast(), layers, resolution, EXTENT, grid);
    }

    private MetricPoint measure() {
        return new MetricPoint(
                epoch,
                network.loss(trainX, trainY),
                network.loss(testX, testY),
                network.accuracy(trainX, trainY),
                network.accuracy(testX, testY));
    }

    private void record(MetricPoint point) {
        history.add(point);
        if (history.size() > maxHistory) {
            // Keep the curve's shape while bounding memory: drop every other point (but keep the newest).
            List<MetricPoint> thinned = new ArrayList<>(history.size() / 2 + 1);
            for (int i = 0; i < history.size() - 1; i += 2) {
                thinned.add(history.get(i));
            }
            thinned.add(history.getLast());
            history = thinned;
        }
    }

    void touch() {
        lastActivity = clock.instant();
    }

    Instant lastActivity() {
        return lastActivity;
    }

    List<SseEmitter> subscribers() {
        return subscribers;
    }

    public String id() {
        return id;
    }

    public TrainingConfig config() {
        return config;
    }

    public Dataset dataset() {
        return dataset;
    }
}
