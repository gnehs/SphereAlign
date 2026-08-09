//! Cross-platform configuration for child processes launched by the desktop app.

use std::ffi::OsStr;
use std::process::Command;

/// Build a command without opening a console window on Windows.
pub(crate) fn silent_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    configure_windows(&mut command);
    command
}

#[cfg(windows)]
fn configure_windows(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    // https://learn.microsoft.com/windows/win32/procthread/process-creation-flags
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn configure_windows(_: &mut Command) {}
