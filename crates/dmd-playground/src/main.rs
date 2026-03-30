use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use dashmap::DashMap;
use dmd_core::{AnalysisOptions, AnalysisReport, ApproximationMethod, analyze_source};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

const INDEX_HTML: &str = include_str!("../assets/index.html");
const APP_JS: &str = include_str!("../assets/app.js");
const STYLES_CSS: &str = include_str!("../assets/styles.css");

#[derive(Debug, Parser)]
#[command(
    name = "dmd-playground",
    about = "Sandboxed playground for symbolic data movement analysis"
)]
struct ServerOptions {
    #[arg(long, default_value_t = 3000)]
    port: u16,

    #[arg(long, default_value_t = 4)]
    max_concurrent: usize,

    #[arg(long, default_value_t = 64 * 1024)]
    max_source_bytes: usize,

    #[arg(long, default_value_t = 30)]
    timeout_seconds: u64,

    #[arg(long, default_value_t = 5_000_000)]
    max_operations_cap: usize,
}

#[derive(Debug, Clone)]
struct PlaygroundConfig {
    max_concurrent: usize,
    max_source_bytes: usize,
    timeout: Duration,
    max_operations_cap: usize,
}

impl From<ServerOptions> for PlaygroundConfig {
    fn from(value: ServerOptions) -> Self {
        Self {
            max_concurrent: value.max_concurrent.max(1),
            max_source_bytes: value.max_source_bytes.max(1024),
            timeout: Duration::from_secs(value.timeout_seconds.max(1)),
            max_operations_cap: value.max_operations_cap.max(10_000),
        }
    }
}

#[derive(Debug, Clone)]
struct AppState {
    tasks: TaskManager,
}

#[derive(Debug, Clone)]
struct TaskManager {
    config: Arc<PlaygroundConfig>,
    limiter: Arc<Semaphore>,
    tasks: Arc<DashMap<Uuid, TaskRecord>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum TaskPhase {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
struct TaskRecord {
    phase: TaskPhase,
    result: Option<AnalysisReport>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CreateTaskRequest {
    source: String,
    #[serde(default = "default_block_size")]
    block_size: usize,
    #[serde(default = "default_num_sets")]
    num_sets: usize,
    max_operations: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct TaskAccepted {
    task_id: Uuid,
    status: TaskPhase,
}

#[derive(Debug, Clone, Serialize)]
struct TaskStatusResponse {
    task_id: Uuid,
    status: TaskPhase,
    result: Option<AnalysisReport>,
    error: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    PayloadTooLarge(String),
    #[error("{0}")]
    NotFound(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
        };
        (status, self.to_string()).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .without_time()
        .init();

    let options = ServerOptions::parse();
    let port = options.port;
    let config = PlaygroundConfig::from(options);
    let state = AppState {
        tasks: TaskManager::new(config),
    };
    let app = router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles))
        .route("/api/tasks", post(create_task))
        .route("/api/tasks/{task_id}", get(task_status))
        .with_state(state)
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/javascript"),
        )],
        APP_JS,
    )
}

async fn styles() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/css; charset=utf-8"),
        )],
        STYLES_CSS,
    )
}

async fn create_task(
    State(state): State<AppState>,
    Json(request): Json<CreateTaskRequest>,
) -> Result<Json<TaskAccepted>, AppError> {
    let accepted = state.tasks.submit(request).await?;
    Ok(Json(accepted))
}

async fn task_status(
    Path(task_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<TaskStatusResponse>, AppError> {
    let snapshot = state
        .tasks
        .snapshot(task_id)
        .ok_or_else(|| AppError::NotFound(format!("unknown task `{task_id}`")))?;
    Ok(Json(snapshot))
}

impl TaskManager {
    fn new(config: PlaygroundConfig) -> Self {
        Self {
            limiter: Arc::new(Semaphore::new(config.max_concurrent)),
            config: Arc::new(config),
            tasks: Arc::new(DashMap::new()),
        }
    }

    async fn submit(&self, request: CreateTaskRequest) -> Result<TaskAccepted, AppError> {
        validate_request(&request, &self.config)?;

        let effective_operations = request
            .max_operations
            .unwrap_or(self.config.max_operations_cap)
            .min(self.config.max_operations_cap);
        let task_id = Uuid::now_v7();
        self.tasks.insert(
            task_id,
            TaskRecord {
                phase: TaskPhase::Queued,
                result: None,
                error: None,
            },
        );

        let manager = self.clone();
        tokio::spawn(async move {
            let Ok(permit) = manager.limiter.clone().acquire_owned().await else {
                manager.fail(task_id, "task limiter closed".to_string());
                return;
            };
            manager.set_phase(task_id, TaskPhase::Running);
            let timeout = manager.config.timeout;
            let source = request.source;
            let options = AnalysisOptions {
                block_size: request.block_size,
                num_sets: request.num_sets,
                max_operations: effective_operations,
                approximation_method: ApproximationMethod::Scale,
            };

            let result = tokio::time::timeout(
                timeout,
                tokio::task::spawn_blocking(move || analyze_source(&source, options)),
            )
            .await;

            match result {
                Ok(Ok(Ok(report))) => manager.complete(task_id, report),
                Ok(Ok(Err(error))) => manager.fail(task_id, error.to_string()),
                Ok(Err(join_error)) => manager.fail(task_id, join_error.to_string()),
                Err(_) => manager.fail(task_id, "analysis timed out".to_string()),
            }

            drop(permit);
        });

        Ok(TaskAccepted {
            task_id,
            status: TaskPhase::Queued,
        })
    }

    fn snapshot(&self, task_id: Uuid) -> Option<TaskStatusResponse> {
        self.tasks.get(&task_id).map(|record| TaskStatusResponse {
            task_id,
            status: record.phase.clone(),
            result: record.result.clone(),
            error: record.error.clone(),
        })
    }

    fn set_phase(&self, task_id: Uuid, phase: TaskPhase) {
        if let Some(mut record) = self.tasks.get_mut(&task_id) {
            record.phase = phase;
        }
    }

    fn complete(&self, task_id: Uuid, report: AnalysisReport) {
        if let Some(mut record) = self.tasks.get_mut(&task_id) {
            record.phase = TaskPhase::Completed;
            record.result = Some(report);
            record.error = None;
        }
    }

    fn fail(&self, task_id: Uuid, error: String) {
        if let Some(mut record) = self.tasks.get_mut(&task_id) {
            record.phase = TaskPhase::Failed;
            record.result = None;
            record.error = Some(error);
        }
    }
}

fn validate_request(
    request: &CreateTaskRequest,
    config: &PlaygroundConfig,
) -> Result<(), AppError> {
    if request.source.len() > config.max_source_bytes {
        return Err(AppError::PayloadTooLarge(format!(
            "source exceeds sandbox limit of {} bytes",
            config.max_source_bytes
        )));
    }
    if request.block_size == 0 {
        return Err(AppError::BadRequest(
            "block_size must be at least one".to_string(),
        ));
    }
    if request.num_sets == 0 {
        return Err(AppError::BadRequest(
            "num_sets must be at least one".to_string(),
        ));
    }
    Ok(())
}

const fn default_block_size() -> usize {
    1
}

const fn default_num_sets() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, sleep};

    const SOURCE: &str = r#"
params N;
array A[N];

for i in 0 .. N {
    read A[0];
}
"#;

    #[tokio::test]
    async fn task_manager_completes_successfully() {
        let manager = TaskManager::new(PlaygroundConfig {
            max_concurrent: 1,
            max_source_bytes: 4096,
            timeout: Duration::from_secs(10),
            max_operations_cap: 1_000_000,
        });
        let accepted = manager
            .submit(CreateTaskRequest {
                source: SOURCE.to_string(),
                block_size: 1,
                num_sets: 1,
                max_operations: Some(200_000),
            })
            .await
            .expect("task submission should succeed");
        for _ in 0..50 {
            let snapshot = manager
                .snapshot(accepted.task_id)
                .expect("task should exist");
            if matches!(snapshot.status, TaskPhase::Completed) {
                assert!(snapshot.result.is_some());
                return;
            }
            sleep(Duration::from_millis(20)).await;
        }
        panic!("task did not complete in time");
    }

    #[tokio::test]
    async fn oversized_source_is_rejected() {
        let manager = TaskManager::new(PlaygroundConfig {
            max_concurrent: 1,
            max_source_bytes: 8,
            timeout: Duration::from_secs(1),
            max_operations_cap: 1_000,
        });
        let error = manager
            .submit(CreateTaskRequest {
                source: SOURCE.to_string(),
                block_size: 1,
                num_sets: 1,
                max_operations: None,
            })
            .await
            .expect_err("request should be rejected");
        assert!(matches!(error, AppError::PayloadTooLarge(_)));
    }

    #[test]
    fn assets_include_monaco_backed_dsl_editor() {
        assert!(INDEX_HTML.contains("source-editor"));
        assert!(INDEX_HTML.contains("source-fallback"));
        assert!(INDEX_HTML.contains("monaco-editor@0.52.2"));
        assert!(INDEX_HTML.contains("katex@0.16.11"));
        assert!(INDEX_HTML.contains("defer src=\"/app.js\""));
        assert!(APP_JS.contains("monaco.languages.register"));
        assert!(APP_JS.contains("katex.render"));
        assert!(APP_JS.contains("CtrlCmd"));
        assert!(APP_JS.contains("array C[M, N];"));
        assert!(APP_JS.contains("read C[i, j];"));
        assert!(APP_JS.contains("write C[i, j];"));
        assert!(STYLES_CSS.contains(".editor-shell"));
        assert!(STYLES_CSS.contains(".guide-grid"));
        assert!(STYLES_CSS.contains(".math-block"));
    }
}
