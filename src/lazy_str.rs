use std::{cell::OnceCell, fmt::Arguments};

/// Creates a [`LazyStr`] from `format_args!` without eagerly allocating a `String`.
#[macro_export]
macro_rules! lazy_str {
    ($($arg:tt)*) => {
        $crate::LazyStr::new(format_args!($($arg)*))
    };
}

/// Lazily formats string arguments on first string access.
pub struct LazyStr<'a> {
    args: Arguments<'a>,
    value: OnceCell<String>,
}

impl<'a> LazyStr<'a> {
    /// Creates a lazily formatted string from prebuilt formatting arguments.
    pub fn new(args: Arguments<'a>) -> Self {
        Self {
            args,
            value: OnceCell::new(),
        }
    }
}

impl AsRef<str> for LazyStr<'_> {
    fn as_ref(&self) -> &str {
        self.value.get_or_init(|| self.args.to_string()).as_str()
    }
}
