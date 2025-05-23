use std::{io::Write, path::Path};

use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Serialize, Deserialize)]
pub enum AnalysisRequest {
    Barvinok {
        source: String,
        approximation: bool,
        infinite_repeat: bool,
        block_size: Option<usize>,
        lower_bounds: Vec<usize>,
    },
    Salt {
        source: String,
    },
}

impl AnalysisRequest {
    pub fn run<P: AsRef<Path>>(&self, analyzer: P, license: String) -> anyhow::Result<String> {
        use hakoniwa::*;
        let mut container = Container::new();
        let analyzer_path = analyzer
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid analyzer path"))?;

        container
            .unshare(Namespace::Cgroup)
            .unshare(Namespace::Ipc)
            .unshare(Namespace::Network)
            .unshare(Namespace::Uts);

        container
            .rootfs("/")
            .devfsmount("/dev")
            .tmpfsmount("/tmp")
            .bindmount_ro(analyzer_path, "/analyzer");

        container
            .setrlimit(Rlimit::Core, 0, 0)
            .setrlimit(Rlimit::Nproc, 8, 8)
            .setrlimit(Rlimit::Rss, 64 * 1024 * 1024, 64 * 1024 * 1024)
            .setrlimit(Rlimit::Nofile, 128, 128);

        {
            use hakoniwa::landlock::*;
            let mut ruleset = Ruleset::default();
            ruleset.restrict(Resource::FS, CompatMode::Enforce);
            ruleset.add_fs_rule("/bin", FsAccess::R | FsAccess::X);
            ruleset.add_fs_rule("/lib", FsAccess::R | FsAccess::X);
            ruleset.add_fs_rule("/lib64", FsAccess::R | FsAccess::X);
            ruleset.add_fs_rule("/usr", FsAccess::R);
            ruleset.add_fs_rule("/dev", FsAccess::R);
            ruleset.add_fs_rule("/tmp", FsAccess::W);
            ruleset.add_fs_rule("/analyzer", FsAccess::R | FsAccess::X);
            container.landlock_ruleset(ruleset);
        }

        let mut command = container.command("/analyzer");
        command.arg("--json");
        let src = match self {
            AnalysisRequest::Barvinok {
                source,
                approximation,
                infinite_repeat,
                block_size,
                lower_bounds,
            } => {
                command.arg("barvinok");
                if *approximation {
                    command.arg("--barvinok-arg=--approximation-method=scale");
                }
                if *infinite_repeat {
                    command.arg("--infinite-repeat");
                }
                if let Some(block_size) = block_size {
                    command.arg(&format!("--block-size={block_size}"));
                }
                for lower_bound in lower_bounds {
                    command.arg(&format!("--symbol-lowerbound={lower_bound}"));
                }
                source
            }
            AnalysisRequest::Salt { source } => {
                command.arg("salt");
                source
            }
        };
        command.env("SYMBOLICA_LICENSE", &license);
        command.env("SYMBOLICA_HIDE_BANNER", "1");
        command.env("RUST_LOG", "info");
        command.env("RUST_BACKTRACE", "0");
        command.stdin(Stdio::MakePipe);
        command.stdout(Stdio::MakePipe);
        command.stderr(Stdio::MakePipe);
        command.wait_timeout(30);
        let mut child = command.spawn()?;
        info!("spawned child process, pid: {}", child.id());
        {
            let mut stdin = child.stdin.take().unwrap();
            stdin.write_all(src.as_bytes())?;
            stdin.flush()?;
            drop(stdin);
            info!("finished writing to stdin, {} bytes", src.len());
        }
        let output = child.wait_with_output()?;
        if output.status.success() {
            let stdout = String::from_utf8(output.stdout)?;
            info!(
                "child process finished successfully with {} bytes",
                stdout.len()
            );
            Ok(stdout)
        } else {
            let stderr = String::from_utf8(output.stderr)?;
            info!(
                "child process failed with reason: {} and {} bytes",
                output.status.reason,
                stderr.len()
            );
            Err(anyhow::anyhow!("Analyzer failed: {}", stderr))
        }
    }
}
