import type { MetricPoint } from '../api/types';

interface Props {
  history: MetricPoint[];
  width?: number;
  height?: number;
}

/** Train/test loss over epochs with test accuracy on a secondary 0–100% scale. */
export function LossChart({ history, width = 380, height = 96 }: Props) {
  if (history.length < 2) {
    return <svg className="chart" width={width} height={height} role="img" aria-label="Loss chart (waiting for data)" />;
  }
  const pad = 2;
  const first = history[0].epoch;
  const last = history[history.length - 1].epoch;
  const span = Math.max(1, last - first);
  const maxLoss = Math.max(0.1, ...history.map((h) => Math.max(h.trainLoss, h.testLoss)));
  const x = (epoch: number) => pad + ((epoch - first) / span) * (width - 2 * pad);
  const y = (v: number, max: number) => height - pad - (Math.min(v, max) / max) * (height - 2 * pad);
  const path = (pick: (h: MetricPoint) => number, max: number) =>
    history.map((h, i) => `${i ? 'L' : 'M'}${x(h.epoch).toFixed(1)},${y(pick(h), max).toFixed(1)}`).join('');

  const latest = history[history.length - 1];
  return (
    <svg
      className="chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Loss curve: train ${latest.trainLoss.toFixed(3)}, test ${latest.testLoss.toFixed(3)} at epoch ${last}`}
    >
      <line x1={0} x2={width} y1={y(0, 1) } y2={y(0, 1)} className="chart__axis" />
      <path d={path((h) => h.testAccuracy, 1)} className="chart__accuracy" />
      <path d={path((h) => h.testLoss, maxLoss)} className="chart__test" />
      <path d={path((h) => h.trainLoss, maxLoss)} className="chart__train" />
    </svg>
  );
}
