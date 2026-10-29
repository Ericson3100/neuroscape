import type { MetricPoint } from '../api/types';
import { LossChart } from './LossChart';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function MetricsStrip({ history }: { history: MetricPoint[] }) {
  const m = history[history.length - 1];
  if (!m) return null;
  return (
    <section className="metrics" aria-label="Training metrics">
      <dl className="metrics__numbers">
        <div>
          <dt>Epoch</dt>
          <dd>{m.epoch.toLocaleString('en-US')}</dd>
        </div>
        <div>
          <dt>Loss <span className="key key--train" /> train</dt>
          <dd>{m.trainLoss.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Loss <span className="key key--test" /> test</dt>
          <dd>{m.testLoss.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Accuracy <span className="key key--acc" /> test</dt>
          <dd className="metrics__accent">{pct(m.testAccuracy)}</dd>
        </div>
      </dl>
      <LossChart history={history} />
    </section>
  );
}
