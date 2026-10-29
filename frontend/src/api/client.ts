import type { ModelSummary, ProblemDetail, SessionResponse, TrainingConfig, TrainingSnapshot } from './types';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly problem: ProblemDetail,
  ) {
    super(problem.detail ?? problem.title ?? `HTTP ${status}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...init?.headers },
  });
  if (!response.ok) {
    const problem = (await response.json().catch(() => ({}))) as ProblemDetail;
    throw new ApiError(response.status, problem);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  createSession: (config: TrainingConfig) => post<SessionResponse>('/api/sessions', config),
  deleteSession: (id: string) =>
    fetch(`/api/sessions/${id}`, { method: 'DELETE', keepalive: true }).catch(() => undefined),
  play: (id: string) => post<TrainingSnapshot>(`/api/sessions/${id}/play`),
  pause: (id: string) => post<TrainingSnapshot>(`/api/sessions/${id}/pause`),
  step: (id: string, epochs = 1) => post<TrainingSnapshot>(`/api/sessions/${id}/step?epochs=${epochs}`),
  reset: (id: string) => post<TrainingSnapshot>(`/api/sessions/${id}/reset`),
  streamUrl: (id: string) => `/api/sessions/${id}/stream`,

  listModels: () => request<ModelSummary[]>('/api/models'),
  saveModel: (sessionId: string, name: string) => post<ModelSummary>('/api/models', { sessionId, name }),
  deleteModel: (id: number) => request<void>(`/api/models/${id}`, { method: 'DELETE' }),
  loadModel: (id: number) => post<SessionResponse>(`/api/models/${id}/load`),
};
