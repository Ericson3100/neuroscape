package dev.ericson.neuroscape.training;

import java.util.List;

/**
 * What the browser needs to render one frame: metrics, every weight, and the model's output sampled
 * on a {@code gridResolution × gridResolution} lattice over {@code [-extent, extent]²} (row-major, y-major).
 */
public record TrainingSnapshot(
        String sessionId,
        boolean running,
        MetricPoint metrics,
        List<LayerParameters> layers,
        int gridResolution,
        double extent,
        float[] grid) {

    /** Parameters connecting layer {@code l} to {@code l+1}: {@code weights[out][in]}, {@code biases[out]}. */
    public record LayerParameters(double[][] weights, double[] biases) {
    }
}
