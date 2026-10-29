import type { Activation, DatasetType, OptimizerType, TrainingConfig } from '../api/types';
import { ArchitectureEditor } from './ArchitectureEditor';
import { Segmented } from './Segmented';

const datasets: { value: DatasetType; label: string }[] = [
  { value: 'SPIRAL', label: 'Spiral' },
  { value: 'CIRCLES', label: 'Circles' },
  { value: 'XOR', label: 'XOR' },
  { value: 'MOONS', label: 'Moons' },
  { value: 'GAUSSIAN', label: 'Blobs' },
];
const activations: { value: Activation; label: string }[] = [
  { value: 'TANH', label: 'tanh' },
  { value: 'RELU', label: 'ReLU' },
  { value: 'LEAKY_RELU', label: 'leaky' },
  { value: 'SIGMOID', label: 'sigmoid' },
];
const optimizers: { value: OptimizerType; label: string }[] = [
  { value: 'SGD', label: 'SGD' },
  { value: 'MOMENTUM', label: 'Momentum' },
  { value: 'ADAM', label: 'Adam' },
];
const learningRates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
const batchSizes = [1, 8, 16, 32, 64];

interface Props {
  config: TrainingConfig;
  onChange: (patch: Partial<TrainingConfig>) => void;
}

export function ControlRail({ config, onChange }: Props) {
  return (
    <div className="rail__section">
      <Segmented label="Dataset" value={config.dataset} options={datasets} onChange={(dataset) => onChange({ dataset })} />

      <div className="field">
        <label className="field__label" htmlFor="noise">
          Noise <span className="field__value">{config.noise.toFixed(2)}</span>
        </label>
        <input
          id="noise"
          type="range"
          min={0}
          max={0.5}
          step={0.01}
          value={config.noise}
          onChange={(e) => onChange({ noise: Number(e.target.value) })}
        />
      </div>

      <div className="field field--row">
        <label className="field__label" htmlFor="samples">Samples</label>
        <select id="samples" value={config.samples} onChange={(e) => onChange({ samples: Number(e.target.value) })}>
          {[200, 400, 800].map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <button type="button" className="link" onClick={() => onChange({ seed: Math.floor(Math.random() * 2 ** 31) })}>
          reseed
        </button>
      </div>

      <ArchitectureEditor layers={config.hiddenLayers} onChange={(hiddenLayers) => onChange({ hiddenLayers })} />
      <Segmented label="Activation" value={config.activation} options={activations} onChange={(activation) => onChange({ activation })} />
      <Segmented label="Optimizer" value={config.optimizer} options={optimizers} onChange={(optimizer) => onChange({ optimizer })} />

      <div className="field field--row">
        <label className="field__label" htmlFor="lr">Learning rate</label>
        <select id="lr" value={config.learningRate} onChange={(e) => onChange({ learningRate: Number(e.target.value) })}>
          {learningRates.map((lr) => (
            <option key={lr} value={lr}>{lr}</option>
          ))}
        </select>
      </div>
      <div className="field field--row">
        <label className="field__label" htmlFor="batch">Batch size</label>
        <select id="batch" value={config.batchSize} onChange={(e) => onChange({ batchSize: Number(e.target.value) })}>
          {batchSizes.map((b) => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
