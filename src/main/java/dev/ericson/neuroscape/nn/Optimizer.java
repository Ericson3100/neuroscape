package dev.ericson.neuroscape.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * Updates parameters in place given their gradients. Implementations may keep per-parameter state,
 * so an optimizer instance belongs to exactly one network.
 */
public sealed interface Optimizer permits Optimizer.Sgd, Optimizer.Momentum, Optimizer.Adam {

    void step(List<double[]> params, List<double[]> grads);

    static Optimizer of(OptimizerType type, double learningRate) {
        return switch (type) {
            case SGD -> new Sgd(learningRate);
            case MOMENTUM -> new Momentum(learningRate, 0.9);
            case ADAM -> new Adam(learningRate, 0.9, 0.999, 1e-8);
        };
    }

    /** Plain gradient descent: {@code θ ← θ − η·g}. */
    final class Sgd implements Optimizer {
        private final double lr;

        public Sgd(double lr) {
            this.lr = lr;
        }

        @Override
        public void step(List<double[]> params, List<double[]> grads) {
            for (int k = 0; k < params.size(); k++) {
                double[] p = params.get(k);
                double[] g = grads.get(k);
                for (int i = 0; i < p.length; i++) {
                    p[i] -= lr * g[i];
                }
            }
        }
    }

    /** Classical momentum: {@code v ← μ·v − η·g; θ ← θ + v}. */
    final class Momentum implements Optimizer {
        private final double lr;
        private final double mu;
        private List<double[]> velocity;

        public Momentum(double lr, double mu) {
            this.lr = lr;
            this.mu = mu;
        }

        @Override
        public void step(List<double[]> params, List<double[]> grads) {
            if (velocity == null) {
                velocity = zerosLike(params);
            }
            for (int k = 0; k < params.size(); k++) {
                double[] p = params.get(k);
                double[] g = grads.get(k);
                double[] v = velocity.get(k);
                for (int i = 0; i < p.length; i++) {
                    v[i] = mu * v[i] - lr * g[i];
                    p[i] += v[i];
                }
            }
        }
    }

    /**
     * Adam (Kingma &amp; Ba, 2015): bias-corrected first and second moment estimates.
     * <pre>
     * m ← β1·m + (1−β1)·g        v ← β2·v + (1−β2)·g²
     * m̂ = m / (1−β1ᵗ)            v̂ = v / (1−β2ᵗ)
     * θ ← θ − η·m̂ / (√v̂ + ε)
     * </pre>
     */
    final class Adam implements Optimizer {
        private final double lr;
        private final double beta1;
        private final double beta2;
        private final double epsilon;
        private List<double[]> m;
        private List<double[]> v;
        private int t;

        public Adam(double lr, double beta1, double beta2, double epsilon) {
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
        }

        @Override
        public void step(List<double[]> params, List<double[]> grads) {
            if (m == null) {
                m = zerosLike(params);
                v = zerosLike(params);
            }
            t++;
            double correction1 = 1 - Math.pow(beta1, t);
            double correction2 = 1 - Math.pow(beta2, t);
            for (int k = 0; k < params.size(); k++) {
                double[] p = params.get(k);
                double[] g = grads.get(k);
                double[] mk = m.get(k);
                double[] vk = v.get(k);
                for (int i = 0; i < p.length; i++) {
                    mk[i] = beta1 * mk[i] + (1 - beta1) * g[i];
                    vk[i] = beta2 * vk[i] + (1 - beta2) * g[i] * g[i];
                    double mHat = mk[i] / correction1;
                    double vHat = vk[i] / correction2;
                    p[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
        }
    }

    private static List<double[]> zerosLike(List<double[]> params) {
        List<double[]> zeros = new ArrayList<>(params.size());
        for (double[] p : params) {
            zeros.add(new double[p.length]);
        }
        return zeros;
    }
}
