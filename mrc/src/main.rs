use std::path::PathBuf;

use palc::Parser;
use serde::Deserialize;
use splines::Key;

#[derive(Debug, Clone, Deserialize)]
struct JsonInput {
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
}

#[derive(Debug, Clone, Deserialize)]
struct MissRatioCurve {
    turning_points: Box<[f64]>,
    miss_ratio: Box<[f64]>,
}

#[derive(Debug, Clone, Parser)]
struct Option {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,
    /// Target cache size
    #[arg(short, long)]
    cache_size: u64,
    /// Target block size
    #[arg(short, long)]
    block_size: u64,
}

fn main() {
    let option = Option::parse();

    // Load the JSON input file
    let json_input: JsonInput = simd_json::from_reader(std::fs::File::open(&option.input).unwrap())
        .expect("Failed to parse JSON input");

    // Process the miss ratio curve
    let turning_points = json_input.miss_ratio_curve.turning_points;
    let miss_ratio = json_input.miss_ratio_curve.miss_ratio;
    let total_count: f64 = json_input
        .total_count
        .trim_end_matches(" R")
        .parse()
        .expect("Invalid total count");
    let miss_count = miss_ratio
        .iter()
        .map(|&v| v * total_count)
        .collect::<Vec<_>>();

    let start_time = std::time::Instant::now();
    let sequence = turning_points
        .iter()
        .zip(miss_count.iter())
        .map(|(k, v)| Key::new(*k, *v, splines::Interpolation::CatmullRom))
        .collect::<Vec<_>>();
    let spline = splines::Spline::from_vec(sequence);
    let target0 = (option.cache_size / option.block_size) as f64 / 2.0;
    let target1 = (option.cache_size / option.block_size) as f64;
    let target2 = (option.cache_size / option.block_size) as f64 * 2.0;
    let sample0 = spline.clamped_sample(target0).unwrap();
    let sample1 = spline.clamped_sample(target1).unwrap();
    let sample2 = spline.clamped_sample(target2).unwrap();
    let elapsed = start_time.elapsed().as_nanos() as f64 / 1E6;
    let program_name = option
        .input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let sample = (sample0 + sample1 + sample2) / 3.0;
    println!("{program_name},{},{:.4}", sample.round(), elapsed);
}
