/**
 * three.js scene: pheromone field, fertile patches, food, trails, bonds, glow and bodies.
 * The torus is shown by drawing every layer in a 3x3 tiling around the camera centre
 * (geometry and materials are shared, so tiles cost only draw calls).
 */
import * as THREE from 'three';
import type { FrameMsg } from '../protocol';
import { P_STRIDE } from '../protocol';
import * as S from './shaders';
import { Trails } from './trails';
import { Camera2D } from './camera';

export interface Layers { pheromone: boolean; food: boolean; patches: boolean; bonds: boolean; trails: boolean; glow: boolean }

type Drawable = THREE.Points | THREE.Mesh | THREE.LineSegments;

export class View {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  cam: Camera2D;
  W: number;
  layers: Layers = { pheromone: true, food: true, patches: true, bonds: true, trails: false, glow: true };
  private tiles: THREE.Group[] = [];
  private pGeo: THREE.BufferGeometry;
  private pBuf: THREE.InterleavedBuffer;
  private foodGeo: THREE.BufferGeometry;
  private patchGeo: THREE.BufferGeometry;
  private bondGeo: THREE.BufferGeometry;
  private pheroTex: THREE.DataTexture;
  private uniforms = { uPxPerUnit: { value: 100 } };
  private groups: Record<keyof Layers | 'bodies', Drawable[]> = { pheromone: [], food: [], patches: [], bonds: [], trails: [], glow: [], bodies: [] };
  private trails: Trails;
  private selGeo = new THREE.BufferGeometry();
  private selMat: THREE.ShaderMaterial;
  private fx: { mesh: THREE.Points; t0: number; dur: number }[] = [];

  constructor(canvas: HTMLCanvasElement, W: number, capacity: number, foodCapacity: number, pheroRes: number, patchRadius: number) {
    this.W = W;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x050811, 1);
    this.cam = new Camera2D(W);

    // particles: one interleaved buffer straight from the worker (x, y, size, r, g, b, signal, aggr)
    this.pBuf = new THREE.InterleavedBuffer(new Float32Array(capacity * P_STRIDE), P_STRIDE);
    this.pBuf.setUsage(THREE.DynamicDrawUsage);
    this.pGeo = new THREE.BufferGeometry();
    this.pGeo.setAttribute('position', new THREE.InterleavedBufferAttribute(this.pBuf, 2, 0));
    this.pGeo.setAttribute('aSize', new THREE.InterleavedBufferAttribute(this.pBuf, 1, 2));
    this.pGeo.setAttribute('aColor', new THREE.InterleavedBufferAttribute(this.pBuf, 3, 3));
    this.pGeo.setAttribute('aSignal', new THREE.InterleavedBufferAttribute(this.pBuf, 1, 6));
    this.pGeo.setAttribute('aAggr', new THREE.InterleavedBufferAttribute(this.pBuf, 1, 7));
    this.pGeo.setDrawRange(0, 0);
    const sphere = new THREE.Sphere(new THREE.Vector3(W / 2, W / 2, 0), W * 2);

    this.foodGeo = dynGeo(foodCapacity * 2, 2);
    this.patchGeo = dynGeo(4096, 2);
    this.bondGeo = dynGeo(capacity * 2 * 4, 2);
    // 2-component positions: give three explicit bounds so it never computes them.
    for (const g of [this.pGeo, this.foodGeo, this.patchGeo, this.bondGeo, this.selGeo]) g.boundingSphere = sphere;

    const res = pheroRes;
    this.pheroTex = new THREE.DataTexture(new Uint8Array(res * res * 4), res, res, THREE.RGBAFormat);
    this.pheroTex.magFilter = THREE.LinearFilter;
    this.pheroTex.minFilter = THREE.LinearFilter;
    this.pheroTex.wrapS = this.pheroTex.wrapT = THREE.RepeatWrapping;
    this.pheroTex.needsUpdate = true;

    const add = THREE.AdditiveBlending;
    const pheroMat = mat(S.pheroVert, S.pheroFrag, add, {
      uTex: { value: this.pheroTex }, uTintA: { value: new THREE.Color(0.1, 0.75, 0.85) },
      uTintB: { value: new THREE.Color(0.95, 0.35, 0.75) }, uAlpha: { value: 0.42 },
    });
    const patchMat = mat(S.dotVert, S.patchFrag, add, {
      ...this.uniforms, uRadius: { value: patchRadius * 3.2 }, uMinPx: { value: 6 },
      uColor: { value: new THREE.Color(0.25, 0.75, 0.4) }, uAlpha: { value: 0.22 },
    });
    const foodMat = mat(S.dotVert, S.foodFrag, add, {
      ...this.uniforms, uRadius: { value: 0.0045 }, uMinPx: { value: 1.4 },
      uColor: { value: new THREE.Color(0.55, 1.0, 0.45) }, uAlpha: { value: 0.75 },
    });
    const bondMat = new THREE.LineBasicMaterial({ color: 0xb8e4ff, transparent: true, opacity: 0.45, blending: add, depthTest: false, depthWrite: false });
    const glowMat = mat(S.glowVert, S.glowFrag, add, { ...this.uniforms, uRadius: { value: 0.009 }, uGlow: { value: 1.0 } });
    const bodyMat = mat(S.bodyVert, S.bodyFrag, THREE.NormalBlending, { ...this.uniforms, uRadius: { value: 0.009 } });

    this.trails = new Trails(this.renderer, this.pGeo, this.uniforms);
    const plane = new THREE.PlaneGeometry(W, W);
    plane.translate(W / 2, W / 2, 0);

    for (let t = 0; t < 9; t++) {
      const g = new THREE.Group();
      g.position.set(((t % 3) - 1) * W, (Math.floor(t / 3) - 1) * W, 0);
      const put = (key: keyof View['groups'], o: Drawable, order: number) => {
        o.frustumCulled = false; o.renderOrder = order; g.add(o); this.groups[key].push(o);
      };
      put('pheromone', new THREE.Mesh(plane, pheroMat), 0);
      put('patches', new THREE.Points(this.patchGeo, patchMat), 1);
      put('food', new THREE.Points(this.foodGeo, foodMat), 2);
      put('bonds', new THREE.LineSegments(this.bondGeo, bondMat), 4);
      put('glow', new THREE.Points(this.pGeo, glowMat), 5);
      put('bodies', new THREE.Points(this.pGeo, bodyMat), 6);
      this.tiles.push(g);
      this.scene.add(g);
    }
    const composite = this.trails.compositeMesh();
    composite.renderOrder = 3;
    this.scene.add(composite);
    this.groups.trails.push(composite);

    this.selGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2), 2));
    this.selMat = mat(S.dotVert, S.ringFrag, add, {
      ...this.uniforms, uRadius: { value: 0.03 }, uMinPx: { value: 14 },
      uColor: { value: new THREE.Color(1.6, 1.6, 1.6) }, uProgress: { value: 0.7 }, uWidth: { value: 0.16 },
    });
    const sel = new THREE.Points(this.selGeo, this.selMat);
    sel.frustumCulled = false; sel.renderOrder = 7; sel.visible = false;
    this.scene.add(sel);
  }

  resize(w: number, h: number): void {
    this.renderer.setSize(w, h, false);
    this.cam.resize(w, h);
    const pr = this.renderer.getPixelRatio();
    this.trails.resize(Math.floor(w * pr), Math.floor(h * pr));
  }

  /** Upload a worker frame into GPU buffers. */
  upload(f: FrameMsg): void {
    (this.pBuf.array as Float32Array).set(f.particles.subarray(0, f.n * P_STRIDE));
    this.pBuf.clearUpdateRanges();
    this.pBuf.addUpdateRange(0, f.n * P_STRIDE);
    this.pBuf.needsUpdate = true;
    this.pGeo.setDrawRange(0, f.n);
    setDyn(this.foodGeo, f.food, f.nFood * 2);
    setDyn(this.patchGeo, f.patches, f.patches.length);
    setDyn(this.bondGeo, f.bonds, f.nBonds * 4);
    if (f.pheromone) {
      (this.pheroTex.image.data as Uint8Array).set(f.pheromone);
      this.pheroTex.needsUpdate = true;
    }
    const sel = this.selGeo.attributes.position as THREE.BufferAttribute;
    const obj = this.scene.children.find((c) => (c as THREE.Points).geometry === this.selGeo)!;
    obj.visible = !!f.inspect;
    if (f.inspect) {
      sel.setXY(0, f.inspect.x, f.inspect.y);
      sel.needsUpdate = true;
      const wrapX = f.inspect.x - Math.floor((f.inspect.x - this.cam.cx + this.W / 2) / this.W) * this.W;
      const wrapY = f.inspect.y - Math.floor((f.inspect.y - this.cam.cy + this.W / 2) / this.W) * this.W;
      obj.position.set(wrapX - f.inspect.x, wrapY - f.inspect.y, 0);
    }
  }

  /** A one-shot expanding ring at world (x, y). */
  burst(x: number, y: number, color: THREE.ColorRepresentation, radius: number, dur = 900): void {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([x, y]), 2));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(x, y, 0), 1);
    const m = mat(S.dotVert, S.ringFrag, THREE.AdditiveBlending, {
      ...this.uniforms, uRadius: { value: radius }, uMinPx: { value: 10 },
      uColor: { value: new THREE.Color(color) }, uProgress: { value: 0 }, uWidth: { value: 0.08 },
    });
    const mesh = new THREE.Points(geo, m);
    mesh.frustumCulled = false; mesh.renderOrder = 8;
    this.scene.add(mesh);
    this.fx.push({ mesh, t0: performance.now(), dur });
  }

  render(now: number): void {
    const c = this.cam;
    this.uniforms.uPxPerUnit.value = c.pxPerUnit * this.renderer.getPixelRatio();
    this.selMat.uniforms.uProgress.value = 0.62 + 0.08 * Math.sin(now / 180);
    for (const key of Object.keys(this.groups) as (keyof View['groups'])[]) {
      const on = key === 'bodies' || this.layers[key as keyof Layers];
      for (const o of this.groups[key]) o.visible = on;
    }
    // Camera centre lives in [0, W) and the view is at most 2W wide, so the fixed 3x3 tiles
    // always cover it; only tiles intersecting the view are drawn.
    const [x0, x1, y0, y1] = c.bounds();
    const visible: THREE.Vector3[] = [];
    for (const g of this.tiles) {
      const gx = g.position.x, gy = g.position.y;
      g.visible = gx < x1 && gx + this.W > x0 && gy < y1 && gy + this.W > y0;
      if (g.visible) visible.push(g.position);
    }
    this.fx = this.fx.filter((e) => {
      const p = (now - e.t0) / e.dur;
      if (p >= 1) { this.scene.remove(e.mesh); e.mesh.geometry.dispose(); (e.mesh.material as THREE.Material).dispose(); return false; }
      (e.mesh.material as THREE.ShaderMaterial).uniforms.uProgress.value = p;
      return true;
    });
    if (this.layers.trails) this.trails.accumulate(c.camera, visible);
    this.renderer.render(this.scene, c.camera);
  }

  clearTrails(): void { this.trails.clear(); }
}

function mat(vert: string, frag: string, blending: THREE.Blending, uniforms: Record<string, THREE.IUniform>): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({ vertexShader: vert, fragmentShader: frag, uniforms, blending, transparent: true, depthTest: false, depthWrite: false });
}

function dynGeo(floats: number, itemSize: number): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const a = new THREE.BufferAttribute(new Float32Array(floats), itemSize);
  a.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute('position', a);
  g.setDrawRange(0, 0);
  return g;
}

function setDyn(g: THREE.BufferGeometry, src: Float32Array, floats: number): void {
  const a = g.attributes.position as THREE.BufferAttribute;
  const n = Math.min(floats, a.array.length);
  (a.array as Float32Array).set(src.subarray(0, n));
  a.clearUpdateRanges();
  a.addUpdateRange(0, n);
  a.needsUpdate = true;
  g.setDrawRange(0, n / a.itemSize);
}
