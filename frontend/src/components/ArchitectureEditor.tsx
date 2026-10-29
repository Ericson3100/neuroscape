const MAX_LAYERS = 6;
const MAX_NEURONS = 12;

interface Props {
  layers: number[];
  onChange: (layers: number[]) => void;
}

/** Hidden-layer widths as a row of steppers; input (2) and output (1) are fixed. */
export function ArchitectureEditor({ layers, onChange }: Props) {
  const set = (index: number, width: number) =>
    onChange(layers.map((w, i) => (i === index ? Math.min(MAX_NEURONS, Math.max(1, width)) : w)));

  return (
    <fieldset className="field">
      <legend className="field__label">
        Hidden layers <span className="field__hint">2 → {layers.length ? layers.join(' → ') : '∅'} → 1</span>
      </legend>
      <div className="arch">
        {layers.map((width, i) => (
          <div className="arch__layer" key={i}>
            <button type="button" aria-label={`Add neuron to layer ${i + 1}`} onClick={() => set(i, width + 1)} disabled={width >= MAX_NEURONS}>
              +
            </button>
            <span className="arch__width" aria-label={`Layer ${i + 1} width`}>{width}</span>
            <button type="button" aria-label={`Remove neuron from layer ${i + 1}`} onClick={() => set(i, width - 1)} disabled={width <= 1}>
              −
            </button>
          </div>
        ))}
        <div className="arch__actions">
          <button type="button" onClick={() => onChange([...layers, 4])} disabled={layers.length >= MAX_LAYERS}>
            + layer
          </button>
          <button type="button" onClick={() => onChange(layers.slice(0, -1))} disabled={layers.length === 0}>
            − layer
          </button>
        </div>
      </div>
    </fieldset>
  );
}
