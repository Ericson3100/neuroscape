import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { Color, InstancedMesh, Matrix4, Object3D, Quaternion, Vector3 } from 'three';
import type { LayerParameters } from '../api/types';
import { signedColor } from '../lib/colors';
import { layerSizes } from '../lib/forward';
import { neuronPositions } from './layout';

interface Edge {
  layer: number;
  from: number;
  to: number;
  start: Vector3;
  end: Vector3;
  mid: Vector3;
  length: number;
  rotation: Quaternion;
}

interface Props {
  layers: LayerParameters[];
  /** Activations of every layer for the probe sample (input first), or null. */
  activations: number[][] | null;
  reducedMotion: boolean;
}

const UP = new Vector3(0, 1, 0);
const LAYER_PERIOD = 0.55; // seconds for a pulse to cross one layer
const REST = 0.7; // pause after the output lights up
const tmpColor = new Color();
const tmpObject = new Object3D();
const tmpMatrix = new Matrix4();
const tmpVec = new Vector3();

const smooth = (t: number) => t * t * (3 - 2 * t);

export function NetworkGraph({ layers, activations, reducedMotion }: Props) {
  const sizes = useMemo(() => layerSizes(layers), [layers]);
  const shapeKey = sizes.join('-');
  const positions = useMemo(() => neuronPositions(sizes), [shapeKey]);
  const edges = useMemo<Edge[]>(() => {
    const list: Edge[] = [];
    for (let l = 0; l < positions.length - 1; l++) {
      positions[l + 1].forEach((end, to) =>
        positions[l].forEach((start, from) => {
          const dir = end.clone().sub(start);
          list.push({
            layer: l,
            from,
            to,
            start,
            end,
            mid: start.clone().add(end).multiplyScalar(0.5),
            length: dir.length(),
            rotation: new Quaternion().setFromUnitVectors(UP, dir.normalize()),
          });
        }),
      );
    }
    return list;
  }, [positions]);
  const neuronCount = sizes.reduce((a, b) => a + b, 0);

  const edgeMesh = useRef<InstancedMesh>(null);
  const pulseMesh = useRef<InstancedMesh>(null);
  const neuronMesh = useRef<InstancedMesh>(null);
  // Displayed weights ease toward the latest snapshot so 12 Hz updates read as continuous motion.
  const shown = useRef<Float32Array>(new Float32Array(0));

  useFrame(({ clock }, delta) => {
    const edgesM = edgeMesh.current;
    const pulses = pulseMesh.current;
    const neurons = neuronMesh.current;
    if (!edgesM || !pulses || !neurons) return;

    if (shown.current.length !== edges.length) shown.current = new Float32Array(edges.length);
    const k = reducedMotion ? 1 : 1 - Math.exp(-delta * 10);
    let maxAbs = 0.5;
    edges.forEach((e, idx) => {
      const target = layers[e.layer].weights[e.to][e.from];
      shown.current[idx] += (target - shown.current[idx]) * k;
      maxAbs = Math.max(maxAbs, Math.abs(target));
    });

    // Pulse timeline: phase in [0, L) walks layer by layer, then rests.
    const transitions = sizes.length - 1;
    const cycle = transitions * LAYER_PERIOD + REST;
    const phase = reducedMotion || !activations ? Infinity : (clock.elapsedTime % cycle) / LAYER_PERIOD;
    const segment = Math.floor(phase);
    const frac = phase - segment;

    edges.forEach((e, idx) => {
      const w = shown.current[idx];
      const m = Math.min(1, Math.abs(w) / maxAbs);
      tmpObject.position.copy(e.mid);
      tmpObject.quaternion.copy(e.rotation);
      const radius = 0.004 + 0.026 * m;
      tmpObject.scale.set(radius, e.length, radius);
      tmpObject.updateMatrix();
      edgesM.setMatrixAt(idx, tmpObject.matrix);
      edgesM.setColorAt(idx, signedColor(w, 0.18 + 0.82 * m, tmpColor));

      // Pulses carry w·a along the edge for the active layer transition.
      let scale = 0;
      if (activations && e.layer === segment) {
        const contribution = w * activations[e.layer][e.from];
        const strength = Math.min(1, Math.abs(contribution) / maxAbs);
        if (strength > 0.06) {
          scale = 0.035 + 0.05 * strength;
          tmpVec.copy(e.start).lerp(e.end, smooth(frac));
          pulses.setColorAt(idx, signedColor(contribution, 1.6 + 2.2 * strength, tmpColor));
        }
      }
      tmpMatrix.makeScale(scale, scale, scale).setPosition(scale ? tmpVec : e.start);
      pulses.setMatrixAt(idx, tmpMatrix);
    });

    // Neurons glow with their activation once the wave has reached their layer.
    let n = 0;
    positions.forEach((layer, l) =>
      layer.forEach((pos, i) => {
        const reached = activations && l <= segment;
        const a = reached ? activations[l][i] : 0;
        const value = l === sizes.length - 1 && reached ? (a - 0.5) * 2 : Math.tanh(a);
        const arrival = reached && l === segment ? Math.max(0, 1 - frac * 2) : 0;
        const brightness = reached ? 0.35 + 1.6 * Math.abs(value) + arrival * 1.2 : 0;
        if (reached) signedColor(value, brightness, tmpColor);
        else tmpColor.setRGB(0.16, 0.17, 0.19);
        const r = 0.1 + (reached ? 0.03 * Math.abs(value) : 0);
        tmpMatrix.makeScale(r, r, r).setPosition(pos);
        neurons.setMatrixAt(n, tmpMatrix);
        neurons.setColorAt(n, tmpColor);
        n++;
      }),
    );

    for (const mesh of [edgesM, pulses, neurons]) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  });

  const output = activations?.[activations.length - 1]?.[0];
  return (
    <group>
      <instancedMesh key={`e-${shapeKey}`} ref={edgeMesh} args={[undefined, undefined, edges.length]} frustumCulled={false}>
        <cylinderGeometry args={[1, 1, 1, 6, 1, true]} />
        <meshBasicMaterial toneMapped={false} transparent opacity={0.9} />
      </instancedMesh>
      <instancedMesh key={`p-${shapeKey}`} ref={pulseMesh} args={[undefined, undefined, edges.length]} frustumCulled={false}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh key={`n-${shapeKey}`} ref={neuronMesh} args={[undefined, undefined, neuronCount]} frustumCulled={false}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      {positions.map((layer, l) => (
        <Html key={l} position={[layer[0].x, layer[layer.length - 1].y - 0.42, 0]} center className="scene-label" zIndexRange={[5, 0]}>
          {l === 0 ? 'input' : l === positions.length - 1 ? 'output' : `${layer.length} · h${l}`}
        </Html>
      ))}
      {positions[0]?.map((pos, i) => (
        <Html key={`x${i}`} position={[pos.x - 0.32, pos.y, 0]} center className="scene-label scene-label--var" zIndexRange={[5, 0]}>
          {i === 0 ? 'x₁' : 'x₂'}
        </Html>
      ))}
      {output !== undefined && positions.length > 0 && (
        <Html position={[positions[positions.length - 1][0].x + 0.5, positions[positions.length - 1][0].y, 0]} center className="scene-label scene-label--var" zIndexRange={[5, 0]}>
          ŷ {output.toFixed(2)}
        </Html>
      )}
    </group>
  );
}
