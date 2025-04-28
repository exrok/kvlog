use crate::{clear_global_span_id, set_global_span_id};
use std::{
    future::Future,
    mem::ManuallyDrop,
    num::NonZeroU64,
    pin::Pin,
    task::{Context, Poll},
};
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct Spanning<T> {
    pub(crate) inner: ManuallyDrop<T>,
    pub(crate) span: NonZeroU64,
}

impl<T: Future> Future for Spanning<T> {
    type Output = T::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let span = self.span;
        set_global_span_id(crate::SpanID { inner: span });
        // Safety: this just the inline code of pin-project-lite
        let inner = unsafe { &mut self.get_unchecked_mut().inner };
        let inner = unsafe { Pin::new_unchecked(inner).map_unchecked_mut(|v| &mut **v) };
        let res = inner.poll(cx);
        clear_global_span_id();
        res
    }
}

impl<T> Drop for Spanning<T> {
    fn drop(&mut self) {
        set_global_span_id(crate::SpanID { inner: self.span });
        unsafe { ManuallyDrop::drop(&mut self.inner) }
        clear_global_span_id();
    }
}

impl<T> Spanning<T> {
    /// Borrows the wrapped type.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Mutably borrows the wrapped type.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consumes the `Spanning`, returning the wrapped type.
    ///
    /// Note that this drops the span.
    pub fn into_inner(self) -> T {
        let this = ManuallyDrop::new(self);
        let inner: *const ManuallyDrop<T> = &this.inner;
        let inner = unsafe { inner.read() };
        ManuallyDrop::into_inner(inner)
    }
}
