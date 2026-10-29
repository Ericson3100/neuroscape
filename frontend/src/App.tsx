import { useEffect, useState } from 'react';
import type { TrainingConfig } from './api/types';
import { ControlRail } from './components/ControlRail';
import { MetricsStrip } from './components/MetricsStrip';
import { SavedModels } from './components/SavedModels';
import { useReducedMotion } from './hooks/useReducedMotion';
import { useSavedModels } from './hooks/useSavedModels';
import { useTrainingSession } from './hooks/useTrainingSession';
import { palette } from './lib/colors';
import { forward } from './lib/forward';
import type { View } from './scene/CameraRig';
import { Scene } from './scene/Scene';

const DEFAULT_CONFIG: TrainingConfig = {
  dataset: 'SPIRAL',
  samples: 400,
  noise: 0.1,
  hiddenLayers: [10, 10, 8],
  activation: 'TANH',
  optimizer: 'ADAM',
  learningRate: 0.01,
  batchSize: 16,
  seed: 7,
};

const views: { value: View; label: string; key: string }[] = [
  { value: 'network', label: 'Network', key: '1' },
  { value: 'overview', label: 'Both', key: '2' },
  { value: 'surface', label: 'Surface', key: '3' },
];

const FEATURE_INTERVAL_MS = 2600;

export default function App() {
  const reducedMotion = useReducedMotion();
  const training = useTrainingSession(DEFAULT_CONFIG);
  const saved = useSavedModels();
  const { session, snapshot, history, running, controls, connection } = training;
  const [view, setView] = useState<View>('overview');
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [featured, setFeatured] = useState(0);

  // When the pointer is elsewhere, keep the network alive by probing a different training point periodically.
  useEffect(() => {
    if (reducedMotion) return;
    const timer = window.setInterval(() => setFeatured((f) => f + 1), FEATURE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [reducedMotion]);

  const featuredSample = session && session.train.length ? session.train[(featured * 37) % session.train.length] : null;
  const probe = hover ?? (featuredSample ? { x: featuredSample.x, y: featuredSample.y } : null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Space') {
        e.preventDefault();
        controls.toggle();
      } else if (e.key === 'ArrowRight') controls.step();
      else if (e.key === 'r' || e.key === 'R') controls.reset();
      else {
        const v = views.find((option) => option.key === e.key);
        if (v) setView(v.value);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [controls]);

  const probeOutput = probe && snapshot ? forward(snapshot.layers, training.config.activation, probe.x, probe.y).at(-1)![0] : null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" aria-hidden />
          <h1>neuroscape</h1>
          <p className="brand__tag">a neural network, written from scratch in Java, learning in front of you</p>
        </div>
        <nav className="views" aria-label="Camera">
          {views.map((v) => (
            <button key={v.value} type="button" aria-pressed={view === v.value} onClick={() => setView(v.value)}>
              {v.label}
              <kbd>{v.key}</kbd>
            </button>
          ))}
        </nav>
        <div className={`status status--${connection}`} title={`Stream ${connection}`}>
          <span className="status__dot" aria-hidden />
          {connection === 'live' ? (running ? 'training' : 'paused') : connection}
        </div>
      </header>

      <aside className="rail" aria-label="Experiment controls">
        <div className="transport">
          <button type="button" className="transport__play" onClick={controls.toggle} disabled={!session} aria-keyshortcuts="Space">
            {running ? 'Pause' : 'Train'}
          </button>
          <button type="button" onClick={controls.step} disabled={!session || running} aria-keyshortcuts="ArrowRight">
            Step
          </button>
          <button type="button" onClick={controls.reset} disabled={!session} aria-keyshortcuts="R">
            Reset
          </button>
        </div>
        <ControlRail config={training.config} onChange={training.updateConfig} />
        <SavedModels
          models={saved.models}
          error={saved.error}
          canSave={!!session}
          onSave={(name) => session && saved.save(session.id, name)}
          onLoad={training.loadModel}
          onDelete={saved.remove}
        />
        <p className="rail__keys">
          <span><kbd>Space</kbd> train/pause</span> · <span><kbd>→</kbd> step</span> · <span><kbd>R</kbd> reset</span> ·{' '}
          <span><kbd>1–3</kbd> camera</span> · <span>hover the surface to probe</span>
        </p>
      </aside>

      <main className="stage">
        {snapshot && session ? (
          <Scene
            snapshot={snapshot}
            activation={training.config.activation}
            train={session.train}
            test={session.test}
            probe={probe}
            onProbe={setHover}
            view={view}
            reducedMotion={reducedMotion}
          />
        ) : (
          <div className="stage__empty">{training.error ?? 'Starting a session…'}</div>
        )}

        <div className="legend" aria-hidden>
          <span><i style={{ background: palette.accent }} /> class 1 · positive weight</span>
          <span><i style={{ background: palette.cool }} /> class 0 · negative weight</span>
          <span><i className="legend__test" /> test point</span>
        </div>

        {probe && probeOutput !== null && (
          <div className="probe" aria-live="polite">
            <span className="probe__label">{hover ? 'Probe' : 'Sample'}</span>
            <span className="probe__coords">({probe.x.toFixed(2)}, {probe.y.toFixed(2)})</span>
            <span className="probe__arrow">→</span>
            <span className="probe__p" style={{ color: probeOutput >= 0.5 ? palette.accent : palette.cool }}>
              p = {probeOutput.toFixed(3)}
            </span>
          </div>
        )}

        {training.error && snapshot && <p className="error error--floating" role="alert">{training.error}</p>}
        <MetricsStrip history={history} />
      </main>
    </div>
  );
}
