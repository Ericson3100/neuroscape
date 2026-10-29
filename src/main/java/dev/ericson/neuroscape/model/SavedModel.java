package dev.ericson.neuroscape.model;

import java.time.Instant;

import jakarta.persistence.Basic;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

import dev.ericson.neuroscape.data.DatasetType;
import dev.ericson.neuroscape.nn.Activation;
import dev.ericson.neuroscape.nn.OptimizerType;

/** A trained network plus the configuration needed to recreate its dataset and keep training. */
@Entity
@Table(name = "saved_model")
public class SavedModel {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 80)
    private String name;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private DatasetType dataset;

    private int samples;
    private double noise;
    private long seed;

    /** Comma-separated hidden layer widths, e.g. {@code "8,8,4"}. */
    @Column(nullable = false, length = 64)
    private String hiddenLayers;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private Activation activation;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private OptimizerType optimizer;

    private double learningRate;
    private int batchSize;
    private int epoch;
    private double testLoss;
    private double testAccuracy;

    @Column(nullable = false)
    private Instant createdAt;

    /** Network weights in {@link dev.ericson.neuroscape.nn.NetworkCodec} format. */
    @Lob
    @Basic(fetch = FetchType.LAZY)
    @Column(nullable = false)
    private byte[] parameters;

    protected SavedModel() {
        // for JPA
    }

    public SavedModel(String name, DatasetType dataset, int samples, double noise, long seed, String hiddenLayers,
            Activation activation, OptimizerType optimizer, double learningRate, int batchSize, int epoch,
            double testLoss, double testAccuracy, Instant createdAt, byte[] parameters) {
        this.name = name;
        this.dataset = dataset;
        this.samples = samples;
        this.noise = noise;
        this.seed = seed;
        this.hiddenLayers = hiddenLayers;
        this.activation = activation;
        this.optimizer = optimizer;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epoch = epoch;
        this.testLoss = testLoss;
        this.testAccuracy = testAccuracy;
        this.createdAt = createdAt;
        this.parameters = parameters;
    }

    public Long getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public DatasetType getDataset() {
        return dataset;
    }

    public int getSamples() {
        return samples;
    }

    public double getNoise() {
        return noise;
    }

    public long getSeed() {
        return seed;
    }

    public String getHiddenLayers() {
        return hiddenLayers;
    }

    public Activation getActivation() {
        return activation;
    }

    public OptimizerType getOptimizer() {
        return optimizer;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpoch() {
        return epoch;
    }

    public double getTestLoss() {
        return testLoss;
    }

    public double getTestAccuracy() {
        return testAccuracy;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public byte[] getParameters() {
        return parameters;
    }
}
