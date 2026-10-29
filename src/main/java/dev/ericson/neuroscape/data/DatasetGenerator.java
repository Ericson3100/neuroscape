package dev.ericson.neuroscape.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/** Deterministic (seeded) generators for the toy datasets. */
public final class DatasetGenerator {

    /** Fraction of samples held out for the test split. */
    public static final double TEST_FRACTION = 0.2;

    private DatasetGenerator() {
    }

    public static Dataset generate(DatasetType type, int samples, double noise, long seed) {
        if (samples < 4) {
            throw new IllegalArgumentException("Need at least 4 samples");
        }
        Random random = new Random(seed);
        List<Sample> points = switch (type) {
            case SPIRAL -> spiral(samples, noise, random);
            case CIRCLES -> circles(samples, noise, random);
            case XOR -> xor(samples, noise, random);
            case GAUSSIAN -> gaussian(samples, noise, random);
            case MOONS -> moons(samples, noise, random);
        };
        Collections.shuffle(points, random);
        int testSize = (int) Math.round(points.size() * TEST_FRACTION);
        return new Dataset(type, points.subList(testSize, points.size()), points.subList(0, testSize));
    }

    private static List<Sample> spiral(int n, double noise, Random r) {
        List<Sample> out = new ArrayList<>(n);
        int half = n / 2;
        for (int label = 0; label < 2; label++) {
            double phase = label * Math.PI;
            for (int i = 0; i < (label == 0 ? half : n - half); i++) {
                double t = (double) i / half;
                double radius = 0.08 + 0.85 * t;
                double angle = 1.75 * t * 2 * Math.PI + phase;
                out.add(new Sample(
                        radius * Math.sin(angle) + jitter(r, noise * 0.5),
                        radius * Math.cos(angle) + jitter(r, noise * 0.5),
                        label));
            }
        }
        return out;
    }

    private static List<Sample> circles(int n, double noise, Random r) {
        List<Sample> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int label = i % 2;
            double radius = label == 1 ? r.nextDouble() * 0.4 : 0.6 + r.nextDouble() * 0.35;
            double angle = r.nextDouble() * 2 * Math.PI;
            out.add(new Sample(
                    radius * Math.cos(angle) + jitter(r, noise * 0.4),
                    radius * Math.sin(angle) + jitter(r, noise * 0.4),
                    label));
        }
        return out;
    }

    private static List<Sample> xor(int n, double noise, Random r) {
        List<Sample> out = new ArrayList<>(n);
        double padding = 0.06;
        for (int i = 0; i < n; i++) {
            double x = uniformAwayFromZero(r, padding);
            double y = uniformAwayFromZero(r, padding);
            int label = x * y > 0 ? 1 : 0;
            out.add(new Sample(x + jitter(r, noise * 0.4), y + jitter(r, noise * 0.4), label));
        }
        return out;
    }

    private static List<Sample> gaussian(int n, double noise, Random r) {
        List<Sample> out = new ArrayList<>(n);
        double spread = 0.12 + noise * 0.6;
        for (int i = 0; i < n; i++) {
            int label = i % 2;
            double cx = label == 1 ? 0.45 : -0.45;
            double cy = label == 1 ? 0.45 : -0.45;
            out.add(new Sample(cx + r.nextGaussian() * spread, cy + r.nextGaussian() * spread, label));
        }
        return out;
    }

    private static List<Sample> moons(int n, double noise, Random r) {
        List<Sample> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int label = i % 2;
            double t = r.nextDouble() * Math.PI;
            double x = label == 0 ? Math.cos(t) : 1 - Math.cos(t);
            double y = label == 0 ? Math.sin(t) : 0.5 - Math.sin(t);
            // Centre and scale the classic moons shape into [-1, 1]².
            out.add(new Sample(
                    (x - 0.5) / 1.6 + jitter(r, noise * 0.3),
                    (y - 0.25) / 1.6 + jitter(r, noise * 0.3),
                    label));
        }
        return out;
    }

    private static double uniformAwayFromZero(Random r, double padding) {
        double v = r.nextDouble() * (1 - padding) + padding;
        return r.nextBoolean() ? v : -v;
    }

    private static double jitter(Random r, double sigma) {
        return sigma == 0 ? 0 : r.nextGaussian() * sigma;
    }
}
