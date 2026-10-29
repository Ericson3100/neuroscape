package dev.ericson.neuroscape.training;

import java.util.Arrays;
import java.util.List;

import dev.ericson.neuroscape.nn.Network;

/** Result of a single forward pass, including every neuron's activation (input layer first). */
public record Prediction(double probability, int label, List<double[]> activations) {

    public static Prediction of(Network network, double x, double y) {
        Network.ForwardPass pass = network.forward(new double[] {x, y});
        double p = pass.output();
        return new Prediction(p, p >= 0.5 ? 1 : 0, Arrays.asList(pass.activations()));
    }
}
