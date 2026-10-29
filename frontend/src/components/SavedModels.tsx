import { useState, type FormEvent } from 'react';
import type { ModelSummary } from '../api/types';

interface Props {
  models: ModelSummary[];
  error: string | null;
  canSave: boolean;
  onSave: (name: string) => void;
  onLoad: (id: number) => void;
  onDelete: (id: number) => void;
}

export function SavedModels({ models, error, canSave, onSave, onLoad, onDelete }: Props) {
  const [name, setName] = useState('');
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    onSave(name.trim());
    setName('');
  };
  return (
    <div className="rail__section">
      <form className="save" onSubmit={submit}>
        <label className="field__label" htmlFor="model-name">Save weights</label>
        <div className="save__row">
          <input id="model-name" value={name} maxLength={80} placeholder="e.g. spiral 8-8-6" onChange={(e) => setName(e.target.value)} />
          <button type="submit" disabled={!canSave || !name.trim()}>Save</button>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {models.length > 0 && (
        <ul className="models">
          {models.map((m) => (
            <li key={m.id}>
              <button type="button" className="models__load" onClick={() => onLoad(m.id)} title="Load into a new session">
                <span className="models__name">{m.name}</span>
                <span className="models__meta">
                  {m.dataset.toLowerCase()} · {[2, ...m.hiddenLayers, 1].join('-')} · {(m.testAccuracy * 100).toFixed(0)}% @ {m.epoch}
                </span>
              </button>
              <button type="button" className="link" aria-label={`Delete ${m.name}`} onClick={() => onDelete(m.id)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
