use parking_lot::Mutex;

/// Lock-protected growable buffer of f32 samples.
///
/// We deliberately use a `Vec` under a short-lived mutex rather than a true
/// SPSC ring (e.g., rtrb): hold-to-talk recordings are bounded by how long
/// the user can press a key, and the realtime callback already memcpys into
/// a stack buffer before grabbing the lock. The lock is held for at most a
/// `Vec::extend_from_slice` of ~1 ms of audio.
///
/// If we ever need to support >60 s utterances, swap this for `rtrb::Producer`.
pub struct RingBuffer {
    inner: Mutex<Vec<f32>>,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(max_samples: usize) -> Self {
        Self {
            inner: Mutex::new(Vec::with_capacity(max_samples.min(1 << 20))),
            capacity: max_samples,
        }
    }

    pub fn push(&self, samples: &[f32]) {
        let mut buf = self.inner.lock();
        let room = self.capacity.saturating_sub(buf.len());
        let n = samples.len().min(room);
        if n > 0 {
            buf.extend_from_slice(&samples[..n]);
        }
        // Excess samples are dropped — this only happens if the user holds
        // the hotkey for longer than `capacity / sample_rate` seconds.
    }

    pub fn drain(&self) -> Vec<f32> {
        let mut buf = self.inner.lock();
        std::mem::take(&mut *buf)
    }

    /// Copy the current contents without consuming them. Used by the
    /// pre-transcription path so we can transcribe what we have so far
    /// while the user is still holding the hotkey.
    pub fn snapshot(&self) -> Vec<f32> {
        self.inner.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}
