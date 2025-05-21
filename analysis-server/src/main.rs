use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use analysis::AnalysisRequest;
use clap::Parser;
use ntex::{
    http::header::HeaderName,
    web::{
        self,
        types::{Json, State},
    },
};

mod analysis;

#[web::post("/analysis")]
async fn analysis_handler(
    state: State<AppState>,
    body: Json<AnalysisRequest>,
) -> impl web::Responder {
    match state.try_acquire() {
        Ok(_guard) => {
            let request = body.into_inner();
            let path = state.analyzer_path.clone();
            let license = state.symbolica_license.clone();
            let result = ntex::rt::spawn_blocking(move || request.run(path, license))
                .await
                .map_err(|e| {
                    tracing::error!("Failed to spawn blocking task: {}", e);
                    anyhow::anyhow!("Failed to spawn blocking task: {}", e)
                })
                .and_then(|x| x);
            match result {
                Ok(output) => {
                    let mut res = web::HttpResponse::Ok().body(output);
                    res.headers_mut().insert(
                        HeaderName::from_static("content-type"),
                        "application/json".parse().unwrap(),
                    );
                    res
                }
                Err(err) => {
                    tracing::warn!("failed to run analysis: {}", err);
                    web::HttpResponse::InternalServerError().body(err.to_string())
                }
            }
        }
        Err(err) => {
            tracing::warn!("rejected request: {}", err);
            web::HttpResponse::ServiceUnavailable().body(err.to_string())
        }
    }
}
#[derive(Parser)]
struct Options {
    /// Path to the analyzer
    #[clap(long, default_value = "analyzer")]
    analyzer_path: PathBuf,
    /// listening address
    #[clap(long, default_value = "0.0.0.0:8080")]
    address: SocketAddr,

    /// symbolica license string
    #[clap(long, env = "SYMBOLICA_LICENSE")]
    symbolica_license: String,

    /// max concurrent requests
    #[clap(long, default_value = "4")]
    max_concurrent_requests: usize,
}
struct AppState {
    analyzer_path: PathBuf,
    symbolica_license: String,
    max_concurrent_requests: usize,
    current_requests: AtomicUsize,
}

struct Guard<'a> {
    current_requests: &'a AtomicUsize,
}
impl AppState {
    fn try_acquire(&self) -> anyhow::Result<Guard> {
        let mut remaining_try = 128;
        while remaining_try > 0 {
            remaining_try -= 1;
            let current = self.current_requests.load(Ordering::SeqCst);
            let proposed = current + 1;
            if proposed > self.max_concurrent_requests {
                std::hint::spin_loop();
                continue;
            }
            if self
                .current_requests
                .compare_exchange(current, proposed, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(Guard {
                    current_requests: &self.current_requests,
                });
            }
        }
        Err(anyhow::anyhow!(
            "Too many concurrent requests: {}",
            self.max_concurrent_requests
        ))
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.current_requests.fetch_sub(1, Ordering::SeqCst);
    }
}

#[ntex::main]
async fn main() -> std::io::Result<()> {
    let options = Options::parse();
    tracing_subscriber::fmt::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    tracing::info!("Starting server...");
    let canonical_path = options.analyzer_path.canonicalize()?;
    web::HttpServer::new(move || {
        web::App::new()
            .state(AppState {
                analyzer_path: canonical_path.clone(),
                symbolica_license: options.symbolica_license.clone(),
                max_concurrent_requests: options.max_concurrent_requests,
                current_requests: AtomicUsize::new(0),
            })
            .service(analysis_handler)
    })
    .bind(options.address)?
    .run()
    .await
}
