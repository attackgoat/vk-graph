//! Test-only, non-owning observation of completed Vulkan destruction.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

/// Observes completed destruction without retaining a Vulkan object.
///
/// Reports are independent of validation layers and the `checked` feature. A report remains
/// incomplete while any owner retains the object, while destruction is in progress, or if
/// destruction is skipped during unwinding. It does not wait for destruction or GPU work.
#[derive(Clone, Debug)]
pub(crate) struct DisposalReport(Arc<AtomicBool>);

impl DisposalReport {
    /// Whether the object's Vulkan destruction call has returned successfully.
    pub(crate) fn is_disposed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Allocate observation state only when a caller requests a report.
#[derive(Default)]
pub(crate) struct DisposalTracker(OnceLock<DisposalReport>);

impl DisposalTracker {
    pub(crate) fn report(&self) -> DisposalReport {
        self.0
            .get_or_init(|| DisposalReport(Arc::new(AtomicBool::new(false))))
            .clone()
    }

    /// Call only after actual destruction, never from an unconditional Drop guard.
    pub(crate) fn complete(&self) {
        if let Some(report) = self.0.get() {
            report.0.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        std::{
            sync::mpsc::{Receiver, Sender, channel},
            thread,
        },
    };

    struct BlockingOwner {
        tracker: DisposalTracker,
        started: Sender<()>,
        release: Receiver<()>,
    }

    impl Drop for BlockingOwner {
        fn drop(&mut self) {
            self.started.send(()).unwrap();
            self.release.recv().unwrap();
            self.tracker.complete();
        }
    }

    #[test]
    fn report_distinguishes_last_owner_release_from_completed_destruction() {
        let (started_tx, started_rx) = channel();
        let (release_tx, release_rx) = channel();
        let owner = Arc::new(std::sync::Mutex::new(BlockingOwner {
            tracker: DisposalTracker::default(),
            started: started_tx,
            release: release_rx,
        }));
        let weak = Arc::downgrade(&owner);
        let report = owner.lock().unwrap().tracker.report();
        let observer = owner.lock().unwrap().tracker.report();
        assert!(!report.is_disposed());
        let worker = thread::spawn(move || drop(owner));
        started_rx.recv().unwrap();
        // Weak/strong counts alone would claim destruction even though Drop is still blocked.
        assert!(weak.upgrade().is_none());
        assert!(!report.is_disposed());
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(report.is_disposed());
        assert!(observer.is_disposed());
    }
}
