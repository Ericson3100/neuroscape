import { useEffect, useMemo, useRef } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { BufferAttribute, Color, DoubleSide, InstancedMesh, Matrix4, Mesh, PlaneGeometry } from 'three';
import type { Sample, TrainingSnapshot } from '../api/types';
import { divergingColor, palette } from '../lib/colors';
import { SURFACE_HEIGHT, SURFACE_SIZE } from './layout';

interface Props {
  snapshot: TrainingSnapshot;
  train: Sample[];
  test: Sample[];
  probe: { x: number; y: number } | null;
  onProbe: (point: { x: number; y: number } | null) => void;
  reducedMotion: boolean;
}

const tmpColor = new Color();
const tmpMatrix = new Matrix4();
const accent = new Color(palette.accent);
const cool = new Color(palette.cool);

/** Height above the p = 0.5 plane for a probability. */
const heightOf = (p: number) => (p - 0.5) * SURFACE_HEIGHT;

/**
 * The model's output over the input plane as a height field: amber peaks are class 1, steel valleys
 * are class 0, and the translucent plane at p = 0.5 slices it along the decision boundary.
 * Plane-local axes: x = input x, y = input y, z = height (the group is rotated to make z point up).
 */
export function DecisionSurface({ snapshot, train, test, probe, onProbe, reducedMotion }: Props) {
  const { gridResolution: res, extent } = snapshot;
  const scale = SURFACE_SIZE / (2 * extent);

  const geometry = useMemo(() => {
    const g = new PlaneGeometry(SURFACE_SIZE, SURFACE_SIZE, res - 1, res - 1);
    g.setAttribute('color', new BufferAttribute(new Float32Array(res * res * 3), 3));
    return g;
  }, [res]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const current = useRef<Float32Array>(new Float32Array(0));
  const surface = useRef<Mesh>(null);

  useFrame((_, delta) => {
    const grid = snapshot.grid;
    if (current.current.length !== grid.length) current.current = Float32Array.from(grid);
    const k = reducedMotion ? 1 : 1 - Math.exp(-delta * 6);
    const pos = geometry.attributes.position as BufferAttribute;
    const col = geometry.attributes.color as BufferAttribute;
    // PlaneGeometry lists vertices top row first (local +y), while the grid's first row is input y = -extent.
    for (let iy = 0; iy < res; iy++) {
      const row = res - 1 - iy;
      for (let ix = 0; ix < res; ix++) {
        const g = row * res + ix;
        const v = iy * res + ix;
        current.current[g] += (grid[g] - current.current[g]) * k;
        const p = current.current[g];
        pos.setZ(v, heightOf(p));
        divergingColor(p, tmpColor);
        col.setXYZ(v, tmpColor.r, tmpColor.g, tmpColor.b);
      }
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geometry.computeVertexNormals();
  });

  const pointsRef = useRef<InstancedMesh>(null);
  const testRef = useRef<InstancedMesh>(null);
  useEffect(() => {
    const place = (mesh: InstancedMesh | null, samples: Sample[], size: number) => {
      if (!mesh) return;
      samples.forEach((s, i) => {
        tmpMatrix.makeScale(size, size, size).setPosition(s.x * scale, s.y * scale, heightOf(s.label));
        mesh.setMatrixAt(i, tmpMatrix);
        mesh.setColorAt(i, s.label === 1 ? accent : cool);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    };
    place(pointsRef.current, train, 0.042);
    place(testRef.current, test, 0.05);
  }, [train, test, scale]);

  const handleMove = (e: ThreeEvent<PointerEvent>) => {
    if (!e.uv) return;
    e.stopPropagation();
    onProbe({ x: -extent + 2 * extent * e.uv.x, y: -extent + 2 * extent * e.uv.y });
  };

  const probeHeight = probe ? heightOf(sample(snapshot, probe.x, probe.y)) : 0;
  const half = SURFACE_SIZE / 2;

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh ref={surface} geometry={geometry} onPointerMove={handleMove} onPointerOut={() => onProbe(null)}>
        <meshStandardMaterial vertexColors side={DoubleSide} roughness={0.55} metalness={0.05} />
      </mesh>

      {/* p = 0.5 plane: where it cuts the surface is the decision boundary. */}
      <mesh raycast={() => null}>
        <planeGeometry args={[SURFACE_SIZE, SURFACE_SIZE]} />
        <meshBasicMaterial color={palette.ink} transparent opacity={0.05} side={DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments raycast={() => null} position={[0, 0, -SURFACE_HEIGHT / 2 - 0.02]}>
        <edgesGeometry args={[new PlaneGeometry(SURFACE_SIZE, SURFACE_SIZE)]} />
        <lineBasicMaterial color={palette.muted} transparent opacity={0.35} />
      </lineSegments>
      <gridHelper
        args={[SURFACE_SIZE, 8, palette.mid, palette.mid]}
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, -SURFACE_HEIGHT / 2 - 0.02]}
        raycast={() => null}
      />

      <instancedMesh key={`tr-${train.length}`} ref={pointsRef} args={[undefined, undefined, train.length]} raycast={() => null}>
        <sphereGeometry args={[1, 14, 14]} />
        <meshStandardMaterial roughness={0.35} emissive={palette.mid} />
      </instancedMesh>
      <instancedMesh key={`te-${test.length}`} ref={testRef} args={[undefined, undefined, test.length]} raycast={() => null}>
        <octahedronGeometry args={[1, 0]} />
        <meshBasicMaterial wireframe toneMapped={false} />
      </instancedMesh>

      {probe && (
        <group position={[probe.x * scale, probe.y * scale, 0]}>
          <mesh position={[0, 0, (probeHeight - SURFACE_HEIGHT / 2) / 2]} rotation={[Math.PI / 2, 0, 0]} raycast={() => null}>
            <cylinderGeometry args={[0.006, 0.006, Math.abs(probeHeight + SURFACE_HEIGHT / 2), 6]} />
            <meshBasicMaterial color={palette.ink} />
          </mesh>
          <mesh position={[0, 0, probeHeight + 0.02]} raycast={() => null}>
            <ringGeometry args={[0.05, 0.075, 32]} />
            <meshBasicMaterial color={palette.ink} side={DoubleSide} toneMapped={false} />
          </mesh>
        </group>
      )}

      <Html position={[0, -half - 0.3, -SURFACE_HEIGHT / 2]} center className="scene-label scene-label--var" zIndexRange={[5, 0]}>x₁ →</Html>
      <Html position={[-half - 0.3, 0, -SURFACE_HEIGHT / 2]} center className="scene-label scene-label--var" zIndexRange={[5, 0]}>x₂ →</Html>
    </group>
  );
}

/** Bilinear sample of the snapshot grid at an input-space point. */
function sample(snapshot: TrainingSnapshot, x: number, y: number): number {
  const { grid, gridResolution: res, extent } = snapshot;
  const fx = Math.min(res - 1, Math.max(0, ((x + extent) / (2 * extent)) * (res - 1)));
  const fy = Math.min(res - 1, Math.max(0, ((y + extent) / (2 * extent)) * (res - 1)));
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(res - 1, x0 + 1), y1 = Math.min(res - 1, y0 + 1);
  const tx = fx - x0, ty = fy - y0;
  const at = (c: number, r: number) => grid[r * res + c];
  return (at(x0, y0) * (1 - tx) + at(x1, y0) * tx) * (1 - ty) + (at(x0, y1) * (1 - tx) + at(x1, y1) * tx) * ty;
}
