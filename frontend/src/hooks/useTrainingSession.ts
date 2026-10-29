import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { MetricPoint, Sample, TrainingConfig, TrainingSnapshot } from '../api/types';

type SessionRequest = { kind: 'create'; config: TrainingConfig } | { kind: 'load'; modelId: number };

export type ConnectionState = 'connecting' | 'live' | 'offline';

export interface SessionState {
  id: string;
  train: Sample[];
  test: Sample[];
}

const MAX_HISTORY = 800;

function appendMetric(history: MetricPoint[], point: MetricPoint): MetricPoint[] {
  const last = history[history.length - 1];
  if (!last || point.epoch < last.epoch) return [point]; // reset or new session
  if (point.epoch === last.epoch) return history;
  const next = [...history, point];
  return next.length > MAX_HISTORY ? next.filter((_, i) => i % 2 === 0 || i === next.length - 1) : next;
}

/**
 * Owns the lifecycle of one server-side training session: creates it whenever the config changes,
 * subscribes to its SSE stream, and deletes it when replaced or when the page goes away.
 */
export function useTrainingSession(initial: TrainingConfig) {
  const [config, setConfig] = useState(initial);
  const [request, setRequest] = useState<SessionRequest>({ kind: 'create', config: initial });
  const [session, setSession] = useState<SessionState | null>(null);
  const [snapshot, setSnapshot] = useState<TrainingSnapshot | null>(null);
  const [history, setHistory] = useState<MetricPoint[]>([]);
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [error, setError] = useState<string | null>(null);
  const keepRunning = useRef(false);
  const configRef = useRef(initial);
  const sessionIdRef = useRef<string | null>(null);

  // 1. (Re)create the session when the request changes; debounce slider drags.
  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(
      async () => {
        try {
          const res = request.kind === 'create' ? await api.createSession(request.config) : await api.loadModel(request.modelId);
          if (cancelled) {
            void api.deleteSession(res.id);
            return;
          }
          setError(null);
          // Replace (and free) the previous server-side session.
          if (sessionIdRef.current) void api.deleteSession(sessionIdRef.current);
          sessionIdRef.current = res.id;
          configRef.current = res.config;
          setConfig(res.config);
          setSession({ id: res.id, train: res.train, test: res.test });
          setSnapshot(res.snapshot);
          setHistory(res.history);
          if (keepRunning.current) void api.play(res.id).then(setSnapshot);
        } catch (e) {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        }
      },
      request.kind === 'create' ? 220 : 0,
    );
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [request]);

  // Free the server-side session when the tab goes away.
  useEffect(() => {
    const onHide = () => {
      if (sessionIdRef.current) void api.deleteSession(sessionIdRef.current);
    };
    window.addEventListener('pagehide', onHide);
    return () => window.removeEventListener('pagehide', onHide);
  }, []);

  // 2. Stream snapshots for the active session.
  useEffect(() => {
    if (!session) return;
    const source = new EventSource(api.streamUrl(session.id));
    setConnection('connecting');
    source.onopen = () => setConnection('live');
    source.onerror = () => setConnection(source.readyState === EventSource.CLOSED ? 'offline' : 'connecting');
    source.addEventListener('snapshot', (event) => {
      const next = JSON.parse((event as MessageEvent<string>).data) as TrainingSnapshot;
      setConnection('live');
      setSnapshot(next);
      setHistory((h) => appendMetric(h, next.metrics));
    });
    return () => source.close();
  }, [session]);

  const updateConfig = useCallback((patch: Partial<TrainingConfig>) => {
    const next = { ...configRef.current, ...patch };
    configRef.current = next;
    setConfig(next);
    setRequest({ kind: 'create', config: next });
  }, []);

  const loadModel = useCallback((modelId: number) => {
    keepRunning.current = false;
    setRequest({ kind: 'load', modelId });
  }, []);

  const withSession = useCallback(
    (action: (id: string) => Promise<TrainingSnapshot>) => {
      if (!session) return;
      action(session.id)
        .then((s) => {
          setSnapshot(s);
          setHistory((h) => appendMetric(h, s.metrics));
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    },
    [session],
  );

  const running = snapshot?.running ?? false;
  const controls = {
    toggle: () => {
      keepRunning.current = !running;
      withSession(running ? api.pause : api.play);
    },
    step: () => withSession((id) => api.step(id, 1)),
    reset: () => withSession(api.reset),
  };

  return { config, updateConfig, loadModel, session, snapshot, history, running, connection, error, controls };
}
