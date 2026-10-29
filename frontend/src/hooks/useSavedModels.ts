import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ModelSummary } from '../api/types';

export function useSavedModels() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.listModels().then(setModels).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(refresh, [refresh]);

  const save = useCallback(
    async (sessionId: string, name: string) => {
      try {
        await api.saveModel(sessionId, name);
        setError(null);
        refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (id: number) => {
      await api.deleteModel(id).catch((e: Error) => setError(e.message));
      refresh();
    },
    [refresh],
  );

  return { models, error, save, remove };
}
