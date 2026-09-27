//! Shared harness for the library test target.
//!
//! Declare the harness before resources so they drop first. Device/instance clones (including
//! clones retained by pools, resources, or background workers) must not outlive the harness:
//! disposal checks enforce this rule at session completion. Wait for pending GPU work and drain
//! background cleanup before ending the session. Checks are independent of the `checked` feature.

use {
    super::disposal::DisposalReport,
    std::{
        collections::BTreeMap,
        ops::Deref,
        panic::catch_unwind,
        sync::{Mutex, MutexGuard, OnceLock, PoisonError},
        thread::panicking,
    },
    vk_graph::driver::{
        DriverError,
        device::{Device, DeviceInfo},
        instance::{Instance, ValidationError},
    },
};

/// Checks enabled for a test session, independently of library-side `checked` assertions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceChecks {
    /// Enable Vulkan validation and enforce exact diagnostic expectations.
    pub validation: bool,
    /// Require device and instance destruction to complete before ending the session.
    pub disposal: bool,
}

impl Default for DeviceChecks {
    fn default() -> Self {
        Self {
            validation: true,
            disposal: true,
        }
    }
}

/// A Vulkan test session that checks validation and disposal through completed teardown.
///
/// Session checks default on in both checked and unchecked library builds.
pub(crate) struct TestDevice<
    'a,
    T = Device,
    F: Fn() -> Vec<ValidationError> = Box<dyn Fn() -> Vec<ValidationError>>,
> {
    guard: Option<MutexGuard<'a, ()>>,
    device: Option<T>,
    validation: F,
    expected: BTreeMap<String, usize>,
    checks: DeviceChecks,
    disposal: Option<(DisposalReport, DisposalReport)>,
}

impl TestDevice<'_> {
    /// Enables both validation and disposal checks in every feature configuration.
    #[allow(dead_code)]
    pub fn new() -> Result<Self, DriverError> {
        Self::with_checks(DeviceChecks::default())
    }

    /// Select checks explicitly. Disabling validation does not require a Vulkan SDK.
    pub fn with_checks(checks: DeviceChecks) -> Result<Self, DriverError> {
        static SESSION_LOCK: Mutex<()> = Mutex::new(());

        let guard = Self::lock_session(&SESSION_LOCK);

        Self::install_logger();

        let device = Device::create(DeviceInfo::builder().debug(checks.validation).build())?;
        let report = Instance::validation_report(&device.physical.instance);
        let disposal = checks.disposal.then(|| {
            (
                Device::disposal_report(&device),
                Instance::disposal_report(&device.physical.instance)
                    .expect("the test session must own its instance"),
            )
        });

        Ok(Self {
            guard: Some(guard),
            device: Some(device),
            validation: Box::new(move || report.as_ref().map_or_else(Vec::new, |r| r.errors())),
            expected: BTreeMap::new(),
            checks,
            disposal,
        })
    }

    fn lock_session(lock: &Mutex<()>) -> MutexGuard<'_, ()> {
        let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);

        // A previous test's panic must not prevent a fresh session with its own report.
        lock.clear_poison();

        guard
    }

    fn install_logger() {
        static INIT: OnceLock<()> = OnceLock::new();

        INIT.get_or_init(|| {
            // Diagnostics are optional; an executable may already have installed a logger.
            let _ = pretty_env_logger::try_init();
        });
    }
}

impl<'a, T, F: Fn() -> Vec<ValidationError>> TestDevice<'a, T, F> {
    // Mock validation sessions do not own real Vulkan objects to observe for disposal.
    #[cfg(test)]
    fn create(
        guard: MutexGuard<'a, ()>,
        validation: F,
        create: impl FnOnce() -> Result<T, DriverError>,
    ) -> Result<Self, DriverError> {
        let device = create()?;

        Ok(Self {
            guard: Some(guard),
            device: Some(device),
            validation,
            expected: BTreeMap::new(),
            checks: DeviceChecks {
                validation: true,
                disposal: false,
            },
            disposal: None,
        })
    }

    /// The checks this session actually enforces.
    #[allow(dead_code)]
    pub fn enabled_checks(&self) -> DeviceChecks {
        self.checks
    }

    /// Whether library-side assertions were compiled in; does not configure this session.
    #[allow(dead_code)]
    pub fn library_assertions_enabled(&self) -> bool {
        cfg!(feature = "checked")
    }

    /// Ends the session, checking disposal and the final validation diagnostics.
    ///
    /// Release all resources and clones and drain background cleanup first. Drop performs the
    /// same checks on normal scope exit. During unwinding it preserves the original panic.
    #[allow(dead_code)]
    pub fn finish(self) {
        drop(self);
    }

    /// Expects exactly this total number of errors with this ID across the entire session.
    /// Expectations may include teardown errors. Zero counts and duplicate IDs are rejected.
    #[allow(dead_code)]
    pub fn expect_validation_error(&mut self, message_id_name: &str, count: usize) {
        assert!(
            self.checks.validation,
            "validation checking is disabled for this session"
        );
        assert!(
            count > 0,
            "expected validation error count must be positive"
        );
        assert!(
            !self.expected.contains_key(message_id_name),
            "duplicate validation error expectation: {message_id_name}"
        );

        self.expected.insert(message_id_name.to_owned(), count);
    }

    /// Checks enabled validation expectations without ending the session.
    /// Callers must wait for pending GPU work first. Disposal is checked only at session end.
    /// All expectations must already be met when calling this method.
    #[allow(dead_code)]
    pub fn assert_valid(&self) {
        if self.checks.validation {
            self.check_errors((self.validation)());
        }
    }

    fn check_errors(&self, errors: Vec<ValidationError>) {
        let mut actual = BTreeMap::new();
        for error in &errors {
            *actual
                .entry(error.message_id_name.as_deref())
                .or_insert(0usize) += 1;
        }

        let expected = self
            .expected
            .iter()
            .map(|(id, count)| (Some(id.as_str()), *count))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(
            actual, expected,
            "Vulkan validation ERRORs did not match expectations (checks={:?}): {errors:#?}",
            self.checks
        );
    }
}

impl<T: std::fmt::Debug, F: Fn() -> Vec<ValidationError>> std::fmt::Debug for TestDevice<'_, T, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestDevice")
            .field("device", &self.device)
            .field("expected", &self.expected)
            .field("checks", &self.checks)
            .field(
                "library_assertions_enabled",
                &self.library_assertions_enabled(),
            )
            .finish_non_exhaustive()
    }
}

impl<T, F: Fn() -> Vec<ValidationError>> Deref for TestDevice<'_, T, F> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.device.as_ref().unwrap()
    }
}

impl<T, F: Fn() -> Vec<ValidationError>> Drop for TestDevice<'_, T, F> {
    fn drop(&mut self) {
        // Taking the device prevents a second drop if its destructor panics.
        if panicking() {
            // Preserve the original test failure even if teardown also panics.
            let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(self.device.take());
            }));
        } else {
            drop(self.device.take());
        }

        // Do not invoke the reporter while unwinding: even a reporter panic must not
        // replace the original failure. The final snapshot includes all teardown errors.
        let disposed = (!panicking())
            .then(|| {
                self.disposal
                    .as_ref()
                    .map(|(device, instance)| (device.is_disposed(), instance.is_disposed()))
            })
            .flatten();
        let errors = (!panicking() && self.checks.validation).then(|| (self.validation)());

        // Snapshot and teardown belong to this session; assertions must not poison the lock.
        drop(self.guard.take());

        if let Some((device, instance)) = disposed {
            assert!(
                device && instance,
                "Vulkan disposal incomplete (checks={:?}): device={device}, instance={instance}; \
                 release resources and device/instance clones and drain background cleanup \
                 before ending the session. Validation cannot cover deferred teardown.",
                self.checks,
            );
        }

        if let Some(errors) = errors {
            self.check_errors(errors);
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::{DeviceChecks, DriverError, TestDevice, ValidationError},
        std::{
            cell::{Cell, RefCell},
            panic::{AssertUnwindSafe, catch_unwind},
            sync::{Mutex, TryLockError, mpsc},
            thread,
        },
    };

    struct OnDrop<F: FnMut()>(F);

    impl<F: FnMut()> Drop for OnDrop<F> {
        fn drop(&mut self) {
            (self.0)();
        }
    }

    fn error(id: Option<&str>) -> ValidationError {
        ValidationError {
            message_id_name: id.map(str::to_owned),
            message: "test validation error".to_owned(),
        }
    }

    #[test]
    fn disabled_validation_rejects_expectations_and_does_not_call_reporter() {
        assert_eq!(
            DeviceChecks::default(),
            DeviceChecks {
                validation: true,
                disposal: true
            }
        );
        let lock = Mutex::new(());
        let drops = Cell::new(0);
        let mut device = TestDevice::create(
            TestDevice::lock_session(&lock),
            || panic!("disabled reporter must not be called"),
            || Ok(OnDrop(|| drops.set(drops.get() + 1))),
        )
        .unwrap();
        device.checks.validation = false;
        assert_eq!(
            device.enabled_checks(),
            DeviceChecks {
                validation: false,
                disposal: false
            }
        );
        assert_eq!(
            device.library_assertions_enabled(),
            cfg!(feature = "checked")
        );
        assert!(catch_unwind(AssertUnwindSafe(|| device.expect_validation_error("A", 1))).is_err());
        device.assert_valid();
        device.finish();
        assert_eq!(drops.get(), 1);
        assert!(lock.try_lock().is_ok());
    }

    #[test]
    fn finish_checks_teardown_once_and_releases_lock_before_asserting() {
        for fail in [false, true] {
            let lock = Mutex::new(());
            let drops = Cell::new(0);
            let snapshots = Cell::new(0);
            let device = TestDevice::create(
                TestDevice::lock_session(&lock),
                || {
                    assert_eq!(drops.get(), 1);
                    snapshots.set(snapshots.get() + 1);
                    if fail {
                        vec![error(Some("teardown"))]
                    } else {
                        Vec::new()
                    }
                },
                || Ok(OnDrop(|| drops.set(drops.get() + 1))),
            )
            .unwrap();
            assert_eq!(
                catch_unwind(AssertUnwindSafe(|| device.finish())).is_err(),
                fail
            );
            assert_eq!(drops.get(), 1);
            assert_eq!(snapshots.get(), 1);
            assert!(lock.try_lock().is_ok());
        }
    }

    #[test]
    fn contending_session_does_not_inherit_validation_errors() {
        let lock = Mutex::new(());
        let first_errors = Mutex::new(Vec::new());
        let first = TestDevice::create(
            TestDevice::lock_session(&lock),
            || first_errors.lock().unwrap().clone(),
            || {
                Ok(OnDrop(|| {
                    first_errors.lock().unwrap().push(error(Some("first")));
                }))
            },
        )
        .unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        thread::scope(|scope| {
            let second = scope.spawn(|| {
                let second_errors = RefCell::new(Vec::new());
                assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                ready_tx.send(()).unwrap();
                let device = TestDevice::create(
                    TestDevice::lock_session(&lock),
                    || second_errors.borrow().clone(),
                    || {
                        assert_eq!(first_errors.lock().unwrap().len(), 1);
                        Ok(())
                    },
                )
                .unwrap();
                drop(device);
            });
            ready_rx.recv().unwrap();
            assert!(catch_unwind(AssertUnwindSafe(|| drop(first))).is_err());
            second.join().unwrap();
        });
        assert!(lock.try_lock().is_ok());
    }

    #[test]
    fn creation_failure_and_unwind_release_and_drop_once() {
        let lock = Mutex::new(());
        let errors = RefCell::new(Vec::new());
        let result = TestDevice::create(
            TestDevice::lock_session(&lock),
            || {
                assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                errors.borrow().clone()
            },
            || {
                assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                errors.borrow_mut().push(error(Some("creation")));
                Err::<(), _>(DriverError::Unsupported)
            },
        );
        assert!(matches!(result, Err(DriverError::Unsupported)));
        assert!(lock.try_lock().is_ok());
        drop(TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap());

        for (panic_in_body, panic_in_drop) in [(true, false), (false, true), (true, true)] {
            let lock = Mutex::new(());
            let errors = RefCell::new(Vec::new());
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                let device = TestDevice::create(
                    TestDevice::lock_session(&lock),
                    || errors.borrow().clone(),
                    || {
                        Ok(OnDrop(|| {
                            drops.set(drops.get() + 1);
                            errors.borrow_mut().push(error(Some("teardown")));
                            if panic_in_drop {
                                panic!("device teardown panic");
                            }
                        }))
                    },
                )
                .unwrap();
                if panic_in_body {
                    panic!("test body panic");
                }
                drop(device);
            }));
            assert_eq!(drops.get(), 1);
            assert_eq!(
                *result.unwrap_err().downcast::<&str>().unwrap(),
                if panic_in_body {
                    "test body panic"
                } else {
                    "device teardown panic"
                }
            );
            assert!(matches!(lock.try_lock(), Err(TryLockError::Poisoned(_))));
            // Recover the lock with an independent report after either original panic.
            let next =
                TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap();
            next.assert_valid();
            drop(next);
            assert!(lock.try_lock().is_ok());
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = TestDevice::create(
                TestDevice::lock_session(&lock),
                || errors.borrow().clone(),
                || -> Result<(), DriverError> {
                    errors.borrow_mut().push(error(Some("creation panic")));
                    panic!("device creation panic");
                },
            );
        }));
        assert_eq!(
            *result.unwrap_err().downcast::<&str>().unwrap(),
            "device creation panic"
        );
        assert!(lock.is_poisoned());
        drop(TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap());
        assert!(lock.try_lock().is_ok());
    }

    #[test]
    fn validation_failures_are_scoped_through_teardown() {
        // No error, creation error, execution error, and teardown error.
        for phase in 0..4 {
            let lock = Mutex::new(());
            let errors = RefCell::new(Vec::new());
            let snapshots = Cell::new(0);
            let drops = Cell::new(0);
            let result = catch_unwind(AssertUnwindSafe(|| {
                let device = TestDevice::create(
                    TestDevice::lock_session(&lock),
                    || {
                        assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                        snapshots.set(snapshots.get() + 1);
                        assert_eq!(drops.get(), 1, "snapshot must follow destruction");
                        errors.borrow().clone()
                    },
                    || {
                        assert_eq!(snapshots.get(), 0, "no process baseline is needed");
                        assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                        if phase == 1 {
                            errors.borrow_mut().push(error(Some("creation")));
                        }
                        Ok(OnDrop(|| {
                            assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                            drops.set(drops.get() + 1);
                            if phase == 3 {
                                errors.borrow_mut().push(error(Some("teardown")));
                            }
                        }))
                    },
                )
                .unwrap();
                if phase == 2 {
                    errors.borrow_mut().push(error(Some("execution")));
                }
                if phase == 0 {
                    return; // Successful early returns must also finish the session.
                }
                drop(device);
            }));
            assert_eq!(drops.get(), 1);
            assert_eq!(snapshots.get(), 1);
            assert_eq!(result.is_err(), phase != 0);
            assert!(
                !lock.is_poisoned(),
                "validation assertions must release the guard first"
            );
            // An earlier session's error is not a failure in this clean session.
            drop(TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap());
        }
    }

    #[test]
    fn explicit_check_keeps_device_and_lock_alive() {
        let lock = Mutex::new(());
        let errors = RefCell::new(Vec::new());
        let drops = Cell::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let device = TestDevice::create(
                TestDevice::lock_session(&lock),
                || errors.borrow().clone(),
                || Ok(OnDrop(|| drops.set(drops.get() + 1))),
            )
            .unwrap();
            device.assert_valid();
            assert_eq!(drops.get(), 0);
            assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
            errors.borrow_mut().push(error(Some("execution")));
            device.assert_valid();
        }));
        assert!(result.is_err());
        assert_eq!(drops.get(), 1);
        drop(TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap());
        assert!(lock.try_lock().is_ok());
    }

    #[test]
    fn expectations_match_exact_total_counts_and_ids() {
        for (ids, expected, valid) in [
            (vec![], vec![], true),
            (
                vec![Some("A"), Some("B"), Some("A")],
                vec![("A", 2), ("B", 1)],
                true,
            ),
            (vec![], vec![("A", 1)], false),
            (vec![Some("A")], vec![("A", 2)], false),
            (vec![Some("A"), Some("A")], vec![("A", 1)], false),
            (vec![Some("B")], vec![("A", 1)], false),
            (vec![Some("a")], vec![("A", 1)], false),
            (vec![Some("A"), Some("B")], vec![("A", 1)], false),
            (vec![None], vec![], false),
            (vec![None], vec![("", 1)], false),
            (vec![Some("A"), None], vec![("A", 1)], false),
        ] {
            let lock = Mutex::new(());
            let errors: Vec<_> = ids.into_iter().map(error).collect();
            let mut device = TestDevice::create(
                TestDevice::lock_session(&lock),
                || errors.clone(),
                || Ok(()),
            )
            .unwrap();
            for (id, count) in expected {
                device.expect_validation_error(id, count);
            }
            // Explicit checks do not consume errors or expectations.
            for _ in 0..2 {
                assert_eq!(
                    catch_unwind(AssertUnwindSafe(|| device.assert_valid())).is_ok(),
                    valid
                );
            }
            assert_eq!(
                catch_unwind(AssertUnwindSafe(|| drop(device))).is_ok(),
                valid
            );
            assert!(lock.try_lock().is_ok());
        }
    }

    #[test]
    fn expectations_include_teardown_errors() {
        for (teardown_id, expected_count, valid) in [
            (Some("A"), 2, true),
            (Some("A"), 1, false),
            (Some("B"), 2, false),
            (None, 2, false),
        ] {
            let lock = Mutex::new(());
            let errors = RefCell::new(vec![error(Some("A"))]);
            let drops = Cell::new(0);
            let mut device = TestDevice::create(
                TestDevice::lock_session(&lock),
                || errors.borrow().clone(),
                || {
                    Ok(OnDrop(|| {
                        drops.set(drops.get() + 1);
                        errors.borrow_mut().push(error(teardown_id));
                    }))
                },
            )
            .unwrap();
            device.expect_validation_error("A", expected_count);
            assert_eq!(
                catch_unwind(AssertUnwindSafe(|| drop(device))).is_ok(),
                valid
            );
            assert_eq!(drops.get(), 1);
            assert!(lock.try_lock().is_ok());
        }
    }

    #[test]
    fn invalid_expectations_are_rejected_without_changing_existing_ones() {
        let lock = Mutex::new(());
        let mut device = TestDevice::create(
            TestDevice::lock_session(&lock),
            || vec![error(Some("A"))],
            || Ok(()),
        )
        .unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| device.expect_validation_error("A", 0))).is_err());
        device.expect_validation_error("A", 1);
        assert!(catch_unwind(AssertUnwindSafe(|| device.expect_validation_error("A", 2))).is_err());
        device.assert_valid();
        drop(device);
        assert!(lock.try_lock().is_ok());
    }

    #[test]
    fn unrelated_reports_and_logger_errors_do_not_affect_session() {
        let lock = Mutex::new(());
        let own_errors = RefCell::new(Vec::new());
        let other_errors = RefCell::new(vec![error(Some("A"))]);
        let mut device = TestDevice::create(
            TestDevice::lock_session(&lock),
            || own_errors.borrow().clone(),
            || Ok(()),
        )
        .unwrap();
        log::error!(target: "unrelated_application", "an ordinary application error");
        device.assert_valid();
        device.expect_validation_error("A", 1);
        // Another report's matching ID cannot satisfy this session's expectation.
        assert!(catch_unwind(AssertUnwindSafe(|| device.assert_valid())).is_err());
        own_errors.borrow_mut().push(error(Some("A")));
        other_errors.borrow_mut().push(error(Some("A")));
        device.assert_valid();
        drop(device);
        // The next session inherits neither the errors nor their expectations.
        drop(TestDevice::create(TestDevice::lock_session(&lock), Vec::new, || Ok(())).unwrap());
        assert_eq!(other_errors.borrow().len(), 2);
    }
}
