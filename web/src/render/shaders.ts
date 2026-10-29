/**
 * GLSL for the map layers. Everything is alpha-blended (no additive light), so it reads
 * the same on a light or a dark background.
 */

/** Particle sprites: radius in world units = aSize * uRadius, never below `minPx` pixels. */
const spriteVert = (radiusExpr: string, minPx: string) => /* glsl */ `
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
  void main() {
    vColor = aColor;
    vSignal = aSignal;
    vAggr = aAggr;
    float r = max(${radiusExpr} * uPxPerUnit, ${minPx});
    vPx = r;
    gl_PointSize = r * 2.0 + 2.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, 0.0, 1.0);
  }
`;

export const bodyVert = spriteVert('aSize * uRadius', '1.8');

/** Flat filled circle with a thin darker outline; the outline turns red while attacking. */
export const bodyFrag = /* glsl */ `
  uniform vec3 uAttack;
  uniform float uOutlineMix;
  varying vec3 vColor;
  varying float vAggr;
  varying float vPx;
  void main() {
    float px = vPx + 1.0;                       // sprite half-size in pixels
    float d = length(gl_PointCoord * 2.0 - 1.0) * px;   // distance from centre in pixels
    float body = 1.0 - smoothstep(vPx - 0.5, vPx + 0.5, d);
    if (body <= 0.0) discard;
    bool attacking = vAggr > 0.3;
    float lw = attacking ? 1.4 : 0.9;           // outline width in pixels
    float outline = smoothstep(vPx - lw - 0.6, vPx - lw + 0.4, d);
    vec3 edge = attacking ? uAttack : mix(vColor, vec3(0.0), uOutlineMix);
    gl_FragColor = vec4(mix(vColor, edge, outline), body);
  }
`;

/** Optional signal display: a thin ring whose opacity follows |signal|. */
export const signalVert = spriteVert('aSize * uRadius * 1.9', '4.0');

export const signalFrag = /* glsl */ `
  uniform vec3 uInk;
  varying float vSignal;
  varying float vPx;
  void main() {
    float px = vPx + 1.0;
    float d = length(gl_PointCoord * 2.0 - 1.0) * px;
    float ring = 1.0 - smoothstep(0.0, 1.0, abs(d - vPx + 1.0));
    float a = ring * smoothstep(0.15, 0.9, abs(vSignal)) * 0.55;
    if (a <= 0.01) discard;
    gl_FragColor = vec4(uInk, a);
  }
`;

/** Food pellets and fertile patches: sprites with a world-space radius and one colour. */
export const dotVert = /* glsl */ `
  uniform float uPxPerUnit;
  uniform float uRadius;
  uniform float uMinPx;
  varying float vPx;
  void main() {
    float r = max(uRadius * uPxPerUnit, uMinPx);
    vPx = r;
    gl_PointSize = r * 2.0 + 1.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, 0.0, 1.0);
  }
`;

export const foodFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uAlpha;
  varying float vPx;
  void main() {
    float d = length(gl_PointCoord * 2.0 - 1.0) * (vPx + 0.5);
    float a = 1.0 - smoothstep(vPx - 0.6, vPx + 0.4, d);
    if (a <= 0.0) discard;
    gl_FragColor = vec4(uColor, a * uAlpha);
  }
`;

/** Fertile patch: a faint flat disc with a soft edge. */
export const patchFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uAlpha;
  void main() {
    float d = length(gl_PointCoord * 2.0 - 1.0);
    float a = uAlpha * (1.0 - smoothstep(0.55, 1.0, d));
    if (a <= 0.002) discard;
    gl_FragColor = vec4(uColor, a);
  }
`;

/** Pheromone field: single-hue heatmap of the summed channels. */
export const pheroVert = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const pheroFrag = /* glsl */ `
  uniform sampler2D uTex;
  uniform vec3 uColor;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    vec4 t = texture2D(uTex, vUv);
    float v = clamp(t.r + t.g, 0.0, 1.0);
    gl_FragColor = vec4(uColor, v * uAlpha);
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

/** Trails are accumulated as intensity; composite them as a translucent colour. */
export const compositeFrag = /* glsl */ `
  uniform sampler2D uTex;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    vec3 c = texture2D(uTex, vUv).rgb;
    float m = max(max(c.r, c.g), c.b);
    if (m <= 0.01) discard;
    gl_FragColor = vec4(c / m, clamp(m, 0.0, 1.0) * uAlpha);
  }
`;

export const trailFrag = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    if (dot(p, p) >= 1.0) discard;
    gl_FragColor = vec4(vColor * 0.35, 1.0);
  }
`;

/** Thin ring (selection marker, intervention feedback). */
export const ringFrag = /* glsl */ `
  uniform vec3 uColor;
  uniform float uProgress;
  uniform float uWidth;
  uniform float uOpacity;
  void main() {
    float d = length(gl_PointCoord * 2.0 - 1.0);
    float a = (1.0 - smoothstep(0.0, uWidth, abs(d - uProgress))) * uOpacity;
    if (a <= 0.01) discard;
    gl_FragColor = vec4(uColor, a);
  }
`;
