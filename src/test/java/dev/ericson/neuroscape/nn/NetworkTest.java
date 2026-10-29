package dev.ericson.neuroscape.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import dev.ericson.neuroscape.data.Dataset;
import dev.ericson.neuroscape.data.DatasetGenerator;
import dev.ericson.neuroscape.data.DatasetType;

class NetworkTest {

    /**
     * Compares back-propagated gradients with central finite differences
     * {@code (L(θ+h) − L(θ−h)) / 2h} for every single parameter.
     */
    @ParameterizedTest
    @EnumSource(value = Activation.class, names = {"TANH", "SIGMOID", "LEAKY_RELU"})
    void backpropMatchesNumericalGradient(Activation activation) {
        Random random = new Random(42);
        Network network = Network.create(2, List.of(5, 4), activation, random);
        double[][] xs = new double[12][];
        int[] ys = new int[12];
        for (int n = 0; n < xs.length; n++) {
            xs[n] = new double[] {random.nextDouble() * 2 - 1, random.nextDouble() * 2 - 1};
            ys[n] = random.nextBoolean() ? 1 : 0;
        }

        List<double[]> analytic = network.gradients(xs, ys).flatten();
        List<double[]> params = network.parameters();
        double h = 1e-5;
        double worst = 0;
        for (int k = 0; k < params.size(); k++) {
            double[] p = params.get(k);
            for (int i = 0; i < p.length; i++) {
                double original = p[i];
                p[i] = original + h;
                double plus = network.loss(xs, ys);
                p[i] = original - h;
                double minus = network.loss(xs, ys);
                p[i] = original;
                double numeric = (plus - minus) / (2 * h);
                double a = analytic.get(k)[i];
                double relError = Math.abs(a - numeric) / Math.max(1e-8, Math.abs(a) + Math.abs(numeric));
                worst = Math.max(worst, relError);
            }
        }
        assertThat(worst).as("worst relative gradient error").isLessThan(1e-6);
    }

    @ParameterizedTest
    @EnumSource(OptimizerType.class)
    void everyOptimizerReducesLossOnSeparableData(OptimizerType type) {
        Dataset data = DatasetGenerator.generate(DatasetType.GAUSSIAN, 200, 0.0, 3);
        Network network = Network.create(2, List.of(4), Activation.TANH, new Random(1));
        Optimizer optimizer = Optimizer.of(type, type == OptimizerType.ADAM ? 0.03 : 0.1);
        double before = network.loss(data.trainFeatures(), data.trainLabels());
        Random shuffle = new Random(2);
        for (int epoch = 0; epoch < 30; epoch++) {
            network.trainEpoch(data.trainFeatures(), data.trainLabels(), 16, optimizer, shuffle);
        }
        assertThat(network.loss(data.trainFeatures(), data.trainLabels())).isLessThan(before * 0.5);
        assertThat(network.accuracy(data.testFeatures(), data.testLabels())).isGreaterThanOrEqualTo(0.95);
    }

    @Test
    void learnsXorWhichNoLinearModelCan() {
        Dataset data = DatasetGenerator.generate(DatasetType.XOR, 400, 0.0, 7);
        Network network = Network.create(2, List.of(8, 8), Activation.TANH, new Random(7));
        Optimizer adam = Optimizer.of(OptimizerType.ADAM, 0.03);
        Random shuffle = new Random(7);
        for (int epoch = 0; epoch < 150; epoch++) {
            network.trainEpoch(data.trainFeatures(), data.trainLabels(), 16, adam, shuffle);
        }
        assertThat(network.accuracy(data.testFeatures(), data.testLabels())).isGreaterThanOrEqualTo(0.95);
    }

    @Test
    void forwardPassExposesEveryLayer() {
        Network network = Network.create(2, List.of(3, 2), Activation.RELU, new Random(0));
        Network.ForwardPass pass = network.forward(new double[] {0.5, -0.5});
        assertThat(pass.activations()).hasNumberOfRows(4);
        assertThat(pass.activations()[1]).hasSize(3);
        assertThat(pass.output()).isBetween(0.0, 1.0);
    }

    @Test
    void rejectsWrongInputSize() {
        Network network = Network.create(2, List.of(3), Activation.TANH, new Random(0));
        assertThatThrownBy(() -> network.forward(new double[] {1, 2, 3}))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void codecRoundTripPreservesPredictions() {
        Network network = Network.create(2, List.of(6, 3), Activation.LEAKY_RELU, new Random(9));
        Network decoded = NetworkCodec.decode(NetworkCodec.encode(network));
        assertThat(decoded.sizes()).containsExactly(2, 6, 3, 1);
        assertThat(decoded.hiddenActivation()).isEqualTo(Activation.LEAKY_RELU);
        for (double x = -1; x <= 1; x += 0.25) {
            double[] input = {x, -x / 2};
            assertThat(decoded.predict(input)).isEqualTo(network.predict(input));
        }
    }

    @Test
    void codecRejectsGarbage() {
        assertThatThrownBy(() -> NetworkCodec.decode(new byte[] {1, 2, 3, 4, 5, 6, 7, 8}))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void sigmoidIsStableForLargeInputs() {
        assertThat(Activation.sigmoid(1000)).isEqualTo(1.0);
        assertThat(Activation.sigmoid(-1000)).isEqualTo(0.0);
        assertThat(Network.crossEntropy(0.0, 1)).isFinite();
    }
}
