/// Mutex wrapper that logs warnings when held for too long in debug builds.
///
/// In debug builds, holding the lock for more than 1 millisecond will emit
/// an error-level log message. In release builds, this behaves identically
/// to [`std::sync::Mutex`].
pub struct Mutex<T: ?Sized> {
    inner: std::sync::Mutex<T>,
}

impl<T: ?Sized + Default> Default for Mutex<T> {
    fn default() -> Self {
        Self {
            inner: std::sync::Mutex::new(Default::default()),
        }
    }
}

#[cfg(debug_assertions)]
struct DropTracker {
    aquired: std::time::Instant,
    caller: &'static std::panic::Location<'static>,
}

#[cfg(debug_assertions)]
impl DropTracker {
    fn new(caller: &'static std::panic::Location<'static>) -> Self {
        Self {
            aquired: std::time::Instant::now(),
            caller,
        }
    }
    #[inline(never)]
    fn log_overheld_if_needed(&self, type_name: &'static str) {
        let time = self.aquired.elapsed();
        if self.aquired.elapsed() < std::time::Duration::from_millis(1) {
            return;
        }
        use crate as kvlog;
        {
            use kvlog::encoding::Encode;
            let mut log = kvlog::global_logger();
            let mut fields = log.encoder.append_now(kvlog::LogLevel::Error);
            self.caller
                .line()
                .encode_log_value_into(fields.dynamic_key("line"));
            self.caller
                .file()
                .encode_log_value_into(fields.dynamic_key("file"));
            kvlog::encoding::Seconds(time.as_secs_f32())
                .encode_log_value_into(fields.dynamic_key("held"));
            type_name.encode_log_value_into(fields.dynamic_key("type"));
            "kvlog".encode_log_value_into(fields.raw_key(15));
            ("Mutex held unexpectedly long").encode_log_value_into(fields.raw_key(0));
            fields.apply_current_span();
            log.poke();
        };
    }
}

/// lock guard returned by [`Mutex::lock`].
///
/// Releases the lock when dropped. In debug builds, logs a warning if the
/// lock was held for more than 1 millisecond.
pub struct MutexGuard<'a, T: ?Sized + 'a> {
    inner: std::sync::MutexGuard<'a, T>,
    #[cfg(debug_assertions)]
    tracker: DropTracker,
}

impl<T> Mutex<T> {
    /// Creates a new mutex wrapping the given value.
    pub const fn new(t: T) -> Mutex<T> {
        Mutex {
            inner: std::sync::Mutex::new(t),
        }
    }
}
impl<T: ?Sized> Mutex<T> {
    /// Acquires the mutex, blocking until it becomes available.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        MutexGuard {
            inner: self.inner.lock().unwrap(),
            #[cfg(debug_assertions)]
            tracker: DropTracker::new(std::panic::Location::caller()),
        }
    }
}

impl<T: ?Sized> std::ops::Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<T: ?Sized> std::ops::DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

#[cfg(debug_assertions)]
impl<'a, T: ?Sized + 'a> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        self.tracker
            .log_overheld_if_needed(std::any::type_name::<T>());
    }
}
