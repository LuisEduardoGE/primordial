/**
 * Motion trails: particles are splatted into a ping-pong render target that fades a little
 * every frame; the accumulated image is composited under the live particles.
 */
import * as THREE from 'three';
import * as S from './shaders';

export class Trails {
  private renderer: THREE.WebGLRenderer;
  private a: THREE.WebGLRenderTarget;
  private b: THREE.WebGLRenderTarget;
  private fadeScene = new THREE.Scene();
  private fadeMat: THREE.ShaderMaterial;
  private dotScene = new THREE.Scene();
  private dots: THREE.Points[] = [];
  private compMat: THREE.ShaderMaterial;
  private quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private lastVersion = -1;

  constructor(renderer: THREE.WebGLRenderer, geo: THREE.BufferGeometry, uniforms: { uPxPerUnit: THREE.IUniform }) {
    this.renderer = renderer;
    const opts = { type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false };
    this.a = new THREE.WebGLRenderTarget(4, 4, opts);
    this.b = new THREE.WebGLRenderTarget(4, 4, opts);
    const quad = new THREE.PlaneGeometry(2, 2);
    this.fadeMat = new THREE.ShaderMaterial({
      vertexShader: S.screenVert, fragmentShader: S.fadeFrag,
      uniforms: { uTex: { value: this.b.texture }, uFade: { value: 0.94 } }, depthTest: false, depthWrite: false,
    });
    const fq = new THREE.Mesh(quad, this.fadeMat);
    fq.frustumCulled = false;
    this.fadeScene.add(fq);
    const dotMat = new THREE.ShaderMaterial({
      vertexShader: S.bodyVert, fragmentShader: S.trailFrag,
      uniforms: { ...uniforms, uRadius: { value: 0.0045 } },
      blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
    });
    for (let t = 0; t < 9; t++) {
      const p = new THREE.Points(geo, dotMat);
      p.frustumCulled = false;
      this.dots.push(p);
      this.dotScene.add(p);
    }
    this.compMat = new THREE.ShaderMaterial({
      vertexShader: S.screenVert, fragmentShader: S.compositeFrag,
      uniforms: { uTex: { value: this.a.texture }, uAlpha: { value: 0.5 } },
      blending: THREE.NormalBlending, transparent: true, depthTest: false, depthWrite: false,
    });
  }

  compositeMesh(): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compMat);
    m.frustumCulled = false;
    return m;
  }

  resize(w: number, h: number): void {
    this.a.setSize(w, h);
    this.b.setSize(w, h);
    this.clear();
  }

  clear(): void {
    const r = this.renderer, prev = r.getRenderTarget();
    for (const t of [this.a, this.b]) { r.setRenderTarget(t); r.clear(); }
    r.setRenderTarget(prev);
  }

  /** Fade the previous accumulation into the next target and splat the current particles. */
  accumulate(camera: THREE.OrthographicCamera & { }, tiles: THREE.Vector3[]): void {
    const r = this.renderer;
    const version = (camera.userData.version as number | undefined) ?? 0;
    if (version !== this.lastVersion) { this.clear(); this.lastVersion = version; }
    [this.a, this.b] = [this.b, this.a];
    this.fadeMat.uniforms.uTex.value = this.b.texture;
    this.compMat.uniforms.uTex.value = this.a.texture;
    this.dots.forEach((d, i) => {
      d.visible = i < tiles.length;
      if (d.visible) d.position.copy(tiles[i]);
    });
    const auto = r.autoClear;
    r.setRenderTarget(this.a);
    r.render(this.fadeScene, this.quadCam);
    r.autoClear = false;
    r.render(this.dotScene, camera);
    r.autoClear = auto;
    r.setRenderTarget(null);
  }
}
