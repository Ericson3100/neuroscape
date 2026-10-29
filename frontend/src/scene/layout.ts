import { Vector3 } from 'three';

/** World-space placement shared by the scene pieces and the camera rig. */
export const NETWORK_ORIGIN = new Vector3(-3.3, 0.35, 0);
export const SURFACE_ORIGIN = new Vector3(3.1, -0.2, 0);

export const NETWORK_WIDTH = 4.4;
export const NETWORK_HEIGHT = 3.4;

/** Side length of the decision surface and its height for p = 0 → 1. */
export const SURFACE_SIZE = 3.8;
export const SURFACE_HEIGHT = 1.3;

export function neuronPositions(sizes: number[]): Vector3[][] {
  const maxNeurons = Math.max(...sizes);
  const spacing = Math.min(0.46, NETWORK_HEIGHT / Math.max(1, maxNeurons - 1));
  return sizes.map((count, l) => {
    const x = sizes.length === 1 ? 0 : -NETWORK_WIDTH / 2 + (l * NETWORK_WIDTH) / (sizes.length - 1);
    return Array.from({ length: count }, (_, i) => new Vector3(x, ((count - 1) / 2 - i) * spacing, 0));
  });
}
