import { Color } from 'three';

/** One accent (amber, class 1 / positive) against a cool counterweight (steel, class 0 / negative). */
export const palette = {
  background: '#0b0c0e',
  ink: '#e9e6df',
  muted: '#8b9098',
  mid: '#1b1e22',
  accent: '#f2a541',
  cool: '#5fa8c4',
} as const;

const mid = new Color(palette.mid);
const accent = new Color(palette.accent);
const cool = new Color(palette.cool);

/** Diverging colour map for a probability: steel at 0, near-black at 0.5, amber at 1. */
export function divergingColor(p: number, target: Color): Color {
  const t = (p - 0.5) * 2;
  const k = Math.pow(Math.min(1, Math.abs(t)), 0.75);
  return target.copy(mid).lerp(t >= 0 ? accent : cool, k);
}

/** Sign-coded colour scaled by magnitude; values above 1 push into bloom range. */
export function signedColor(value: number, brightness: number, target: Color): Color {
  return target.copy(value >= 0 ? accent : cool).multiplyScalar(brightness);
}
