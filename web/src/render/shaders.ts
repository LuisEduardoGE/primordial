/** GLSL for the particle, glow, food, patch, pheromone, trail and effect layers. */

/** Shared vertex shader for sprites whose radius is given in world units (aSize * uRadius). */
const spriteVert = (extra: string, radiusExpr: string, minPx: string) => /* glsl */ `
  attribute float aSize;
  attribute vec3 aColor;
  attribute float aSignal;
  attribute float aAggr;
  uniform float uPxPerUnit;
  uniform float uRadius;
  varying vec3 vColor;
  varying float vSignal;
  varying float vAggr;
  varying float vPx;
  ${extra}
  void main() {
    vColor = aColor;
    vSignal = aSignal;
    vAggr = aAggr;
    float r = max(${radiusExpr} * uPxPerUnit, ${minPx});
    vPx = r;
    gl_PointSize = r * 2.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, 0.0, 1.0);
  }
`;

export const bodyVert = spriteVert('', 'aSize * uRadius', '1.6');

/** Soft cell body: lit interior, bright membrane, red rim while attacking. */
export const bodyFrag = /* glsl */ `
  varying vec3 vColor;
  varying float vSignal;
  varying float vAggr;
  varying float vPx;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = length(p);
    float aa = 1.5 / max(vPx, 1.0);
    float edge = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    if (edge <= 0.0) discard;
    vec2 l = p - vec2(-0.35, 0.35);
    float light = 0.72 + 0.38 * (1.0 - clamp(length(l) * 0.75, 0.0, 1.0));
    vec3 col = vColor * light;
    float membrane = smoothstep(0.62, 0.95, d);
    col = mix(col, vColor * 1.35 + 0.08, membrane * 0.55);
    float nucleus = 1.0 - smoothstep(0.0, 0.32, length(p - vec2(0.08, -0.06)));
    col = mix(col, vColor * 0.45, nucleus * 0.35 * step(4.0, vPx));
    float attack = smoothstep(0.25, 0.7, vAggr);
    float rim = smoothstep(0.58, 0.8, d);
    col = mix(col, vec3(1.0, 0.18, 0.16), attack * rim);
    gl_FragColor = vec4(col, edge * 0.96);
  }
`;

export const glowVert = spriteVert('', 'aSize * uRadius * (2.6 + 2.2 * abs(aSignal))', '5.0');

/** Additive halo; brighter for loud signallers, reddish for attackers. */
export const glowFrag = /* glsl */ `
  uniform float uGlow;
  varying vec3 vColor;
  varying float vSignal;
  varying float vAggr;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(p, p);
    if (d2 >= 1.0) discard;
    float fall = exp(-d2 * 4.0) - 0.018;
    float s = abs(vSignal);
    vec3 col = mix(vColor, vec3(1.0, 0.3, 0.25), smoothstep(0.3, 0.8, vAggr) * 0.6);
    float a = uGlow * (0.10 + 0.55 * s * s) * fall;
    gl_FragColor = vec4(col * a, 1.0);
  }
`;

/** Food pellets and fertile patches: plain sprites with a world-space radius and a fixed tint. */
export const dotVert = /* glsl */ `
  uniform float uPxPerUnit;
  uniform float uRadius;
  uniform float uMinPx;
  varying float vPx;
  void main() {
    float r = max(uRadius * uPxPerUnit, uMinPx);
    vPx = r;
    gl_PointSize = r * 2.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, 0.0, 1.0);
  }
`;

export const foodFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uAlpha;
  varying float vPx;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = length(p);
    if (d >= 1.0) discard;
    float core = 1.0 - smoothstep(0.25, 0.55, d);
    float halo = (1.0 - d) * 0.35;
    float a = uAlpha * (core + halo);
    gl_FragColor = vec4(uColor * a, 1.0);
  }
`;

export const patchFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(p, p);
    if (d2 >= 1.0) discard;
    float a = uAlpha * exp(-d2 * 3.0) * (1.0 - d2);
    gl_FragColor = vec4(uColor * a, 1.0);
  }
`;

/** World-sized plane sampling the 2-channel pheromone texture with two tints. */
export const pheroVert = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const pheroFrag = /* glsl */ `
  uniform sampler2D uTex;
  uniform vec3 uTintA;
  uniform vec3 uTintB;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    vec4 t = texture2D(uTex, vUv);
    vec3 col = uTintA * t.r + uTintB * t.g;
    gl_FragColor = vec4(col * uAlpha, 1.0);
  }
`;

/** Full-screen quads (trail fade + trail composite). */
export const screenVert = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const fadeFrag = /* glsl */ `
  uniform sampler2D uTex;
  uniform float uFade;
  varying vec2 vUv;
  void main() {
    vec4 c = texture2D(uTex, vUv) * uFade;
    gl_FragColor = vec4(max(c.rgb - 0.004, 0.0), 1.0);
  }
`;

export const compositeFrag = /* glsl */ `
  uniform sampler2D uTex;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    gl_FragColor = vec4(texture2D(uTex, vUv).rgb * uAlpha, 1.0);
  }
`;

/** Tiny trail dots. */
export const trailFrag = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(p, p);
    if (d2 >= 1.0) discard;
    gl_FragColor = vec4(vColor * (1.0 - d2) * 0.5, 1.0);
  }
`;

/** Expanding shockwave ring (meteor / food drop / selection). */
export const ringFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uProgress;
  uniform float uWidth;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = length(p);
    float r = uProgress;
    float a = (1.0 - smoothstep(0.0, uWidth, abs(d - r))) * (1.0 - uProgress * uProgress);
    if (a <= 0.003) discard;
    gl_FragColor = vec4(uColor * a, 1.0);
  }
`;
