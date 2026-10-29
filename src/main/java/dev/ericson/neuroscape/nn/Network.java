package dev.ericson.neuroscape.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A fully-connected feed-forward network for binary classification, written from scratch.
 *
 * <p>Layer {@code l} computes {@code z = W·a + b} followed by the hidden activation; the final layer
 * is a single sigmoid unit trained with binary cross-entropy, which makes the output gradient
 * collapse to {@code dL/dz = p - y}.
 */
public final class Network {

    private static final double EPS = 1e-12;

    private final int[] sizes;
    private final Activation hidden;
    /** weights[l][out][in] connects layer l to layer l+1. */
    private final double[][][] weights;
    /** biases[l][out] belongs to layer l+1. */
    private final double[][] biases;

    public Network(int[] sizes, Activation hidden, double[][][] weights, double[][] biases) {
        if (sizes.length < 2 || sizes[sizes.length - 1] != 1) {
            throw new IllegalArgumentException("Network needs an input layer and a single output unit");
        }
        if (weights.length != sizes.length - 1 || biases.length != sizes.length - 1) {
            throw new IllegalArgumentException("Parameter count does not match layer sizes");
        }
        for (int l = 0; l < weights.length; l++) {
            if (weights[l].length != sizes[l + 1] || biases[l].length != sizes[l + 1]) {
                throw new IllegalArgumentException("Layer " + l + " has wrong output size");
            }
            for (double[] row : weights[l]) {
                if (row.length != sizes[l]) {
                    throw new IllegalArgumentException("Layer " + l + " has wrong input size");
                }
            }
        }
        this.sizes = sizes.clone();
        this.hidden = hidden;
        this.weights = weights;
        this.biases = biases;
    }

    /** Builds a randomly initialised network (He init for ReLU-family, Xavier otherwise; zero biases). */
    public static Network create(int inputs, List<Integer> hiddenLayers, Activation hidden, Random random) {
        int[] sizes = new int[hiddenLayers.size() + 2];
        sizes[0] = inputs;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            sizes[i + 1] = hiddenLayers.get(i);
        }
        sizes[sizes.length - 1] = 1;

        double[][][] w = new double[sizes.length - 1][][];
        double[][] b = new double[sizes.length - 1][];
        for (int l = 0; l < w.length; l++) {
            int fanIn = sizes[l];
            int fanOut = sizes[l + 1];
            double scale = hidden.prefersHeInit() && l < w.length - 1
                    ? Math.sqrt(2.0 / fanIn)
                    : Math.sqrt(2.0 / (fanIn + fanOut));
            w[l] = new double[fanOut][fanIn];
            b[l] = new double[fanOut];
            for (double[] row : w[l]) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = random.nextGaussian() * scale;
                }
            }
        }
        return new Network(sizes, hidden, w, b);
    }

    /** Activations of every layer, including the input (index 0) and the output (last). */
    public record ForwardPass(double[][] preActivations, double[][] activations) {
        public double output() {
            return activations[activations.length - 1][0];
        }
    }

    public ForwardPass forward(double[] input) {
        if (input.length != sizes[0]) {
            throw new IllegalArgumentException("Expected " + sizes[0] + " inputs, got " + input.length);
        }
        int layers = sizes.length;
        double[][] z = new double[layers][];
        double[][] a = new double[layers][];
        a[0] = input.clone();
        z[0] = input.clone();
        for (int l = 0; l < layers - 1; l++) {
            boolean outputLayer = l == layers - 2;
            double[][] w = weights[l];
            double[] zl = new double[sizes[l + 1]];
            double[] al = new double[sizes[l + 1]];
            for (int j = 0; j < zl.length; j++) {
                double sum = biases[l][j];
                double[] row = w[j];
                for (int i = 0; i < row.length; i++) {
                    sum += row[i] * a[l][i];
                }
                zl[j] = sum;
                al[j] = outputLayer ? Activation.sigmoid(sum) : hidden.apply(sum);
            }
            z[l + 1] = zl;
            a[l + 1] = al;
        }
        return new ForwardPass(z, a);
    }

    public double predict(double[] input) {
        return forward(input).output();
    }

    /** Mean binary cross-entropy over the given samples. */
    public double loss(double[][] xs, int[] ys) {
        double total = 0;
        for (int n = 0; n < xs.length; n++) {
            double p = predict(xs[n]);
            total += crossEntropy(p, ys[n]);
        }
        return total / xs.length;
    }

    public double accuracy(double[][] xs, int[] ys) {
        int correct = 0;
        for (int n = 0; n < xs.length; n++) {
            int predicted = predict(xs[n]) >= 0.5 ? 1 : 0;
            if (predicted == ys[n]) {
                correct++;
            }
        }
        return xs.length == 0 ? 0 : (double) correct / xs.length;
    }

    static double crossEntropy(double p, int y) {
        double clipped = Math.min(Math.max(p, EPS), 1 - EPS);
        return y == 1 ? -Math.log(clipped) : -Math.log(1 - clipped);
    }

    public Gradients gradients(double[][] xs, int[] ys) {
        int[] all = new int[xs.length];
        Arrays.setAll(all, i -> i);
        return gradients(xs, ys, all, 0, all.length);
    }

    /**
     * Back-propagation: average gradient of the loss over samples {@code indices[from..to)}.
     */
    public Gradients gradients(double[][] xs, int[] ys, int[] indices, int from, int to) {
        Gradients g = Gradients.zerosLike(weights, biases);
        int count = to - from;
        for (int k = from; k < to; k++) {
            int n = indices[k];
            ForwardPass pass = forward(xs[n]);
            double[][] z = pass.preActivations();
            double[][] a = pass.activations();

            // Output delta for sigmoid + cross-entropy.
            double[] delta = {pass.output() - ys[n]};
            for (int l = weights.length - 1; l >= 0; l--) {
                double[][] dW = g.weights()[l];
                double[] dB = g.biases()[l];
                double[] prev = a[l];
                for (int j = 0; j < delta.length; j++) {
                    dB[j] += delta[j];
                    double[] dRow = dW[j];
                    for (int i = 0; i < prev.length; i++) {
                        dRow[i] += delta[j] * prev[i];
                    }
                }
                if (l == 0) {
                    break;
                }
                // Propagate: delta_prev = (W^T delta) ⊙ f'(z_prev)
                double[] next = new double[sizes[l]];
                double[][] w = weights[l];
                for (int i = 0; i < next.length; i++) {
                    double sum = 0;
                    for (int j = 0; j < delta.length; j++) {
                        sum += w[j][i] * delta[j];
                    }
                    next[i] = sum * hidden.derivative(z[l][i]);
                }
                delta = next;
            }
        }
        g.scale(1.0 / count);
        return g;
    }

    /** One pass over the data in shuffled mini-batches. Returns nothing; metrics are computed separately. */
    public void trainEpoch(double[][] xs, int[] ys, int batchSize, Optimizer optimizer, Random random) {
        int[] order = new int[xs.length];
        Arrays.setAll(order, i -> i);
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        List<double[]> params = parameters();
        for (int from = 0; from < order.length; from += batchSize) {
            int to = Math.min(from + batchSize, order.length);
            Gradients g = gradients(xs, ys, order, from, to);
            optimizer.step(params, g.flatten());
        }
    }

    /**
     * Live views of all trainable parameter arrays (weight rows then the bias vector, per layer).
     * Mutating them mutates the network — this is what optimizers update in place.
     */
    public List<double[]> parameters() {
        List<double[]> params = new ArrayList<>();
        for (int l = 0; l < weights.length; l++) {
            params.addAll(Arrays.asList(weights[l]));
            params.add(biases[l]);
        }
        return params;
    }

    public int[] sizes() {
        return sizes.clone();
    }

    public Activation hiddenActivation() {
        return hidden;
    }

    /** Defensive deep copy of the weights, safe to hand out to serializers. */
    public double[][][] weightsCopy() {
        double[][][] copy = new double[weights.length][][];
        for (int l = 0; l < weights.length; l++) {
            copy[l] = new double[weights[l].length][];
            for (int j = 0; j < weights[l].length; j++) {
                copy[l][j] = weights[l][j].clone();
            }
        }
        return copy;
    }

    public double[][] biasesCopy() {
        double[][] copy = new double[biases.length][];
        for (int l = 0; l < biases.length; l++) {
            copy[l] = biases[l].clone();
        }
        return copy;
    }

    public Network copy() {
        return new Network(sizes, hidden, weightsCopy(), biasesCopy());
    }
}
