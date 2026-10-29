import type { Activation, LayerParameters } from '../api/types';

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

const activations: Record<Activation, (z: number) => number> = {
  TANH: Math.tanh,
  RELU: (z) => Math.max(0, z),
  LEAKY_RELU: (z) => (z > 0 ? z : 0.01 * z),
  SIGMOID: sigmoid,
};

/**
 * Client-side mirror of the Java forward pass, used to animate activations for the probe point
 * without a network round-trip on every mouse move. Returns every layer, input first.
 */
export function forward(layers: LayerParameters[], activation: Activation, x: number, y: number): number[][] {
  const f = activations[activation];
  const out: number[][] = [[x, y]];
  let a = out[0];
  layers.forEach((layer, l) => {
    const isOutput = l === layers.length - 1;
    const next = layer.weights.map((row, j) => {
      let z = layer.biases[j];
      for (let i = 0; i < row.length; i++) z += row[i] * a[i];
      return isOutput ? sigmoid(z) : f(z);
    });
    out.push(next);
    a = next;
  });
  return out;
}

export const layerSizes = (layers: LayerParameters[]) =>
  layers.length === 0 ? [] : [layers[0].weights[0].length, ...layers.map((l) => l.biases.length)];
