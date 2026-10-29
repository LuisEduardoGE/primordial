/** Seeded PRNG (sfc32 seeded via splitmix32) with uniform, normal and integer draws. */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  private spare = 0;
  private hasSpare = false;

  constructor(seed: number) {
    let s = seed >>> 0;
    const split = () => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = split();
    this.b = split();
    this.c = split();
    this.d = split();
    for (let i = 0; i < 12; i++) this.u32();
  }

  u32(): number {
    const t = (((this.a + this.b) | 0) + this.d) | 0;
    this.d = (this.d + 1) | 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) | 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.c = (this.c + t) | 0;
    return t >>> 0;
  }

  /** Uniform in [0, 1). */
  rand(): number {
    return this.u32() / 4294967296;
  }

  /** Standard normal (Box-Muller, caches the second draw). */
  randn(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u = 0;
    while (u <= 1e-12) u = this.rand();
    const v = this.rand();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    this.hasSpare = true;
    return r * Math.cos(2 * Math.PI * v);
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.rand() * n);
  }

  /** In-place Fisher-Yates shuffle of the first `n` entries. */
  shuffle(arr: Int32Array | number[], n = arr.length): void {
    for (let i = n - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }
}
