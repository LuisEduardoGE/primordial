/** ~100 ms multiply-add micro-benchmark (shaped like the brain's inner loop), in MMAC/s. */
export function bench(): number {
  const a = new Float32Array(400), b = new Float32Array(16);
  for (let i = 0; i < 400; i++) a[i] = (i * 0.618) % 1;
  let reps = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < 100) {
    for (let r = 0; r < 500; r++) {
      for (let i = 0; i < 25; i++) {
        const x = a[i];
        for (let h = 0; h < 16; h++) b[h] += x * a[i * 16 + h];
      }
    }
    reps += 500;
  }
  return (reps * 400) / ((performance.now() - t0) * 1000) + b[0] * 0;
}
