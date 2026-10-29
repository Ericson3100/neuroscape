package dev.ericson.neuroscape.data;

/** A labelled point in the input plane. */
public record Sample(double x, double y, int label) {

    public double[] features() {
        return new double[] {x, y};
    }
}
