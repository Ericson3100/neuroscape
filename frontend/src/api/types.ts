export type DatasetType = 'SPIRAL' | 'CIRCLES' | 'XOR' | 'GAUSSIAN' | 'MOONS';
export type Activation = 'TANH' | 'RELU' | 'LEAKY_RELU' | 'SIGMOID';
export type OptimizerType = 'SGD' | 'MOMENTUM' | 'ADAM';

export interface TrainingConfig {
  dataset: DatasetType;
  samples: number;
  noise: number;
  hiddenLayers: number[];
  activation: Activation;
  optimizer: OptimizerType;
  learningRate: number;
  batchSize: number;
  seed: number;
}

export interface Sample {
  x: number;
  y: number;
  label: number;
}

export interface MetricPoint {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
}

export interface LayerParameters {
  /** weights[out][in] */
  weights: number[][];
  biases: number[];
}

export interface TrainingSnapshot {
  sessionId: string;
  running: boolean;
  metrics: MetricPoint;
  layers: LayerParameters[];
  gridResolution: number;
  extent: number;
  /** Row-major output probabilities, row = input y (ascending), column = input x (ascending). */
  grid: number[];
}

export interface SessionResponse {
  id: string;
  config: TrainingConfig;
  train: Sample[];
  test: Sample[];
  history: MetricPoint[];
  snapshot: TrainingSnapshot;
}

export interface ModelSummary {
  id: number;
  name: string;
  dataset: DatasetType;
  hiddenLayers: number[];
  activation: Activation;
  optimizer: OptimizerType;
  learningRate: number;
  epoch: number;
  testLoss: number;
  testAccuracy: number;
  createdAt: string;
}

export interface ProblemDetail {
  title?: string;
  detail?: string;
  status?: number;
  errors?: Record<string, string>;
}
