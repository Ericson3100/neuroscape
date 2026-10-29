package dev.ericson.neuroscape.nn;

/**
 * Element-wise activation functions for hidden layers.
 * The output layer always uses a sigmoid (binary classification with cross-entropy loss).
 */
public enum Activation {

    TANH {
        @Override
        public double apply(double z) {
            return Math.tanh(z);
        }

        @Override
        public double derivative(double z) {
            double t = Math.tanh(z);
            return 1.0 - t * t;
        }
    },

    RELU {
        @Override
        public double apply(double z) {
            return z > 0 ? z : 0;
        }

        @Override
        public double derivative(double z) {
            return z > 0 ? 1 : 0;
        }
    },

    LEAKY_RELU {
        @Override
        public double apply(double z) {
            return z > 0 ? z : LEAKY_SLOPE * z;
        }

        @Override
        public double derivative(double z) {
            return z > 0 ? 1 : LEAKY_SLOPE;
        }
    },

    SIGMOID {
        @Override
        public double apply(double z) {
            return sigmoid(z);
        }

        @Override
        public double derivative(double z) {
            double s = sigmoid(z);
            return s * (1.0 - s);
        }
    };

    private static final double LEAKY_SLOPE = 0.01;

    public abstract double apply(double z);

    /** Derivative with respect to the pre-activation {@code z}. */
    public abstract double derivative(double z);

    /** Numerically stable logistic function. */
    public static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    /** Whether He initialisation (vs. Xavier/Glorot) suits this activation. */
    boolean prefersHeInit() {
        return this == RELU || this == LEAKY_RELU;
    }
}
