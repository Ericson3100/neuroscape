import { Canvas } from '@react-three/fiber';
import { Bloom, EffectComposer, Vignette } from '@react-three/postprocessing';
import type { Activation, Sample, TrainingSnapshot } from '../api/types';
import { palette } from '../lib/colors';
import { forward } from '../lib/forward';
import { CameraRig, type View } from './CameraRig';
import { DecisionSurface } from './DecisionSurface';
import { NetworkGraph } from './NetworkGraph';
import { NETWORK_ORIGIN, SURFACE_ORIGIN } from './layout';

interface Props {
  snapshot: TrainingSnapshot;
  activation: Activation;
  train: Sample[];
  test: Sample[];
  probe: { x: number; y: number } | null;
  onProbe: (p: { x: number; y: number } | null) => void;
  view: View;
  reducedMotion: boolean;
}

export function Scene({ snapshot, activation, train, test, probe, onProbe, view, reducedMotion }: Props) {
  const activations = probe ? forward(snapshot.layers, activation, probe.x, probe.y) : null;
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 3.6, 10.6], fov: 38, near: 0.1, far: 80 }}
      gl={{ antialias: true, preserveDrawingBuffer: true }}
      aria-label="3D view of the neural network and its decision surface"
    >
      <color attach="background" args={[palette.background]} />
      <fog attach="fog" args={[palette.background, 14, 30]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[5, 9, 6]} intensity={1.4} />
      <directionalLight position={[-6, 3, -4]} intensity={0.35} color={palette.cool} />

      <group position={NETWORK_ORIGIN}>
        <NetworkGraph layers={snapshot.layers} activations={activations} reducedMotion={reducedMotion} />
      </group>
      <group position={SURFACE_ORIGIN}>
        <DecisionSurface
          snapshot={snapshot}
          train={train}
          test={test}
          probe={probe}
          onProbe={onProbe}
          reducedMotion={reducedMotion}
        />
      </group>

      <CameraRig view={view} reducedMotion={reducedMotion} />
      <EffectComposer multisampling={4}>
        <Bloom mipmapBlur intensity={0.9} luminanceThreshold={0.85} luminanceSmoothing={0.2} />
        <Vignette offset={0.25} darkness={0.55} />
      </EffectComposer>
    </Canvas>
  );
}
