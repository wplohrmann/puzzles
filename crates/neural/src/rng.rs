//! Tiny deterministic PRNG. Xorshift64 — fine for weight init, dream
//! sampling, and negative-sample selection. Not cryptographic.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state.
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa precision is plenty.
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32
    }

    /// Uniform f32 in [-a, a).
    pub fn uniform_centered(&mut self, a: f32) -> f32 {
        2.0 * a * (self.next_f32() - 0.5)
    }

    /// Uniform usize in [0, n).
    pub fn gen_range(&mut self, n: usize) -> usize {
        if n == 0 { return 0; }
        (self.next_u64() % n as u64) as usize
    }

    /// Standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        // Avoid u==0 → ln(0).
        let mut u: f32;
        loop {
            u = self.next_f32();
            if u > 1e-8 { break; }
        }
        let v = self.next_f32();
        let r = (-2.0 * u.ln()).sqrt();
        let theta = std::f32::consts::TAU * v;
        r * theta.cos()
    }
}
