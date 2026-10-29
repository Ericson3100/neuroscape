package dev.ericson.neuroscape.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Gradient of the loss with respect to every weight and bias, shaped like the network. */
public record Gradients(double[][][] weights, double[][] biases) {

    static Gradients zerosLike(double[][][] weights, double[][] biases) {
        double[][][] w = new double[weights.length][][];
        double[][] b = new double[biases.length][];
        for (int l = 0; l < weights.length; l++) {
            w[l] = new double[weights[l].length][weights[l][0].length];
            b[l] = new double[biases[l].length];
        }
        return new Gradients(w, b);
    }

    void scale(double factor) {
        for (int l = 0; l < weights.length; l++) {
            for (double[] row : weights[l]) {
                for (int i = 0; i < row.length; i++) {
                    row[i] *= factor;
                }
            }
            for (int j = 0; j < biases[l].length; j++) {
                biases[l][j] *= factor;
            }
        }
    }

    /** Same ordering as {@link Network#parameters()}. */
    public List<double[]> flatten() {
        List<double[]> flat = new ArrayList<>();
        for (int l = 0; l < weights.length; l++) {
            flat.addAll(Arrays.asList(weights[l]));
            flat.add(biases[l]);
        }
        return flat;
    }
}
