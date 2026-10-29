package dev.ericson.neuroscape.nn;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Compact binary format for persisting a trained network:
 * {@code magic, version, activation, layerCount, sizes..., then weights and biases as doubles}.
 */
public final class NetworkCodec {

    private static final int MAGIC = 0x4E534350; // "NSCP"
    private static final int VERSION = 1;

    private NetworkCodec() {
    }

    public static byte[] encode(Network network) {
        var bytes = new ByteArrayOutputStream();
        try (var out = new DataOutputStream(bytes)) {
            int[] sizes = network.sizes();
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeUTF(network.hiddenActivation().name());
            out.writeInt(sizes.length);
            for (int size : sizes) {
                out.writeInt(size);
            }
            double[][][] w = network.weightsCopy();
            double[][] b = network.biasesCopy();
            for (int l = 0; l < w.length; l++) {
                for (double[] row : w[l]) {
                    for (double value : row) {
                        out.writeDouble(value);
                    }
                }
                for (double value : b[l]) {
                    out.writeDouble(value);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bytes.toByteArray();
    }

    public static Network decode(byte[] data) {
        try (var in = new DataInputStream(new ByteArrayInputStream(data))) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) {
                throw new IllegalArgumentException("Not a neuroscape network blob");
            }
            Activation activation = Activation.valueOf(in.readUTF());
            int[] sizes = new int[in.readInt()];
            for (int i = 0; i < sizes.length; i++) {
                sizes[i] = in.readInt();
            }
            double[][][] w = new double[sizes.length - 1][][];
            double[][] b = new double[sizes.length - 1][];
            for (int l = 0; l < w.length; l++) {
                w[l] = new double[sizes[l + 1]][sizes[l]];
                b[l] = new double[sizes[l + 1]];
                for (double[] row : w[l]) {
                    for (int i = 0; i < row.length; i++) {
                        row[i] = in.readDouble();
                    }
                }
                for (int j = 0; j < b[l].length; j++) {
                    b[l][j] = in.readDouble();
                }
            }
            return new Network(sizes, activation, w, b);
        } catch (IOException e) {
            throw new IllegalArgumentException("Corrupted network blob", e);
        }
    }
}
