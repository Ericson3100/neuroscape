import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import { CameraControls } from '@react-three/drei';
import { NETWORK_ORIGIN, SURFACE_ORIGIN } from './layout';

export type View = 'overview' | 'network' | 'surface';

type Shot = { position: [number, number, number]; target: [number, number, number] };

const shots: Record<View, Shot> = {
  overview: { position: [0, 2.9, 9.0], target: [0, -0.05, 0] },
  network: {
    position: [NETWORK_ORIGIN.x + 0.6, NETWORK_ORIGIN.y + 0.9, 6.2],
    target: [NETWORK_ORIGIN.x, NETWORK_ORIGIN.y - 0.1, 0],
  },
  surface: {
    position: [SURFACE_ORIGIN.x + 2.6, SURFACE_ORIGIN.y + 4.2, 4.6],
    target: [SURFACE_ORIGIN.x, SURFACE_ORIGIN.y, 0],
  },
};

/** Smooth, interruptible camera moves between named shots (instant when motion is reduced). */
export function CameraRig({ view, reducedMotion }: { view: View; reducedMotion: boolean }) {
  const controls = useRef<CameraControls>(null);
  const aspect = useThree((s) => s.size.width / s.size.height);

  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    const { position, target } = shots[view];
    // Pull back on narrow viewports so both halves of the overview stay in frame.
    const pull = view === 'overview' ? Math.max(1, 1.6 / aspect) : Math.max(1, 1.1 / aspect);
    const p = position.map((v, i) => target[i] + (v - target[i]) * pull) as [number, number, number];
    void c.setLookAt(...p, ...target, !reducedMotion);
  }, [view, aspect, reducedMotion]);

  return (
    <CameraControls
      ref={controls}
      makeDefault
      minDistance={2.5}
      maxDistance={22}
      smoothTime={reducedMotion ? 0 : 0.6}
      maxPolarAngle={Math.PI * 0.62}
    />
  );
}
