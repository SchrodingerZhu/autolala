#![feature(float_gamma)]
#[cfg(feature = "charming")]
use charming::{
    Chart,
    component::{Axis, Title},
    element::{AxisType, Tooltip, Trigger},
    series::Line,
};
use core::f64;
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "plotters")]
use plotters::{coord::Shift, prelude::*};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{DiscreteCDF, Poisson};

#[derive(Serialize, Deserialize)]
pub struct MissRatioCurve {
    miss_ratio: Box<[f64]>,
    turning_points: Box<[f64]>,
}

#[derive(Debug, Clone, Copy)]
pub enum SkewDecay {
    // Exponential: 1 + (alpha - 1) * exp(-k * t)
    Exponential { k: f64 },
    // Gaussian:  f(alpha, t) = 1 + (alpha - 1) * exp(-(t / tau) ^ 2)
    Gaussian { tau: f64 },
    // Rational:  f(alpha, t) = 1 + (alpha - 1) / (1 + (t / tau) ^ p)
    Rational { tau: f64, p: f64 },
    // Logistic: f(alpha, t) = 1 + (alpha - 1) * (2 / (1 + exp(k * t)))
    Logistic { k: f64 },
    // Constant: f(alpha, t) = alpha
    Constant,
}

impl std::str::FromStr for SkewDecay {
    type Err = String;

    // Accepted formats (case-insensitive, spaces ignored around commas):
    // - "exp,<k>" or "exponential,<k>"
    // - "gaussian,<tau>" or "gauss,<tau>"
    // - "rational,<tau>,<p>" or "rat,<tau>,<p>"
    // - "logistic,<k>"
    // - "constant" or "const" (no parameter; an optional trailing number is ignored)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("SkewDecay: empty input".into());
        }

        let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
        if parts.is_empty() {
            return Err("SkewDecay: failed to split input".into());
        }

        let kind = parts[0].to_ascii_lowercase();
        let vals = &parts[1..];

        let parse_f64 = |idx: usize, what: &str| -> Result<f64, String> {
            let raw = vals
                .get(idx)
                .ok_or_else(|| format!("SkewDecay: expected {} parameter", what))?;
            raw.parse::<f64>()
                .map_err(|_| format!("SkewDecay: invalid {}: '{}'", what, raw))
        };

        match kind.as_str() {
            "exp" | "exponential" => {
                let k = parse_f64(0, "k")?;
                Ok(SkewDecay::Exponential { k })
            }
            "gaussian" | "gauss" => {
                let tau = parse_f64(0, "tau")?;
                Ok(SkewDecay::Gaussian { tau })
            }
            "rational" | "rat" => {
                let tau = parse_f64(0, "tau")?;
                let p = parse_f64(1, "p")?;
                Ok(SkewDecay::Rational { tau, p })
            }
            "logistic" => {
                let k = parse_f64(0, "k")?;
                Ok(SkewDecay::Logistic { k })
            }
            "constant" | "const" => Ok(SkewDecay::Constant),
            other => Err(format!(
                "SkewDecay: unknown kind '{}'. Expected one of exp, gaussian, rational, logistic, constant",
                other
            )),
        }
    }
}

impl SkewDecay {
    pub fn value(&self, alpha: f64, t: f64) -> f64 {
        match self {
            SkewDecay::Exponential { k } => 1.0 + (alpha - 1.0) * (-k * t).exp(),
            SkewDecay::Gaussian { tau } => 1.0 + (alpha - 1.0) * (-(t / tau).powi(2)).exp(),
            SkewDecay::Rational { tau, p } => 1.0 + (alpha - 1.0) / (1.0 + (t / tau).powf(*p)),
            SkewDecay::Logistic { k } => 1.0 + (alpha - 1.0) * (2.0 / (1.0 + (k * t).exp())),
            SkewDecay::Constant => alpha,
        }
    }
}

impl MissRatioCurve {
    pub fn compute_assoc(&self, associativity: usize, skewness: f64, decay: SkewDecay) -> Self {
        let len = self.turning_points.len();
        let rd = &self.turning_points;
        // Step 1: Calculate RD distribution (q_j values) from miss ratios
        let mut rd_portions = vec![0.0; len];
        for i in 0..len {
            if i == 0 {
                rd_portions[i] = 1.0 - self.miss_ratio[i];
            } else {
                let tmp = 1.0 - self.miss_ratio[i];
                rd_portions[i] = tmp - (1.0 - self.miss_ratio[i - 1]);
            }
        }

        let max_tp_usize = rd.last().unwrap().ceil() as usize;
        let cache_sizes: Vec<_> = (associativity..(max_tp_usize + associativity))
            .step_by(1)
            .collect();

        // Calculate all miss ratios in parallel and show progress
        let pb = ProgressBar::new(cache_sizes.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .tick_chars("◐◓◑◒ "),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().unwrap().get())
            .build()
            .unwrap();

        let parallel_results: Vec<(f64, f64)> = pool.install(|| {
            cache_sizes
                .par_iter()
                .map_with(pb.clone(), |pb, &cache_size| {
                    let mut miss_ratio = 0.0;
                    let number_of_sets = cache_size as f64 / associativity as f64;

                    for (r, w) in rd.iter().copied().zip(rd_portions.iter().copied()) {
                        if r == 0.0 {
                            continue;
                        }
                        if cache_size == associativity && cache_size as f64 <= r {
                            miss_ratio += w;
                            continue;
                        }
                        let skewness = decay.value(skewness, r);
                        let lambda = r.min(r * skewness / number_of_sets);
                        let pos = Poisson::new(lambda).unwrap();
                        let hit_prob = pos.cdf(associativity as u64 - 1) as f64;
                        miss_ratio += w * (1.0 - hit_prob);
                    }

                    pb.inc(1);
                    (miss_ratio, cache_size as f64)
                })
                .collect()
        });

        pb.finish_and_clear();

        // Create the final vectors with initial values
        let mut new_miss_ratio = parallel_results
            .iter()
            .map(|(ratio, _)| *ratio)
            .collect::<Vec<_>>();
        let mut new_turning_points = parallel_results
            .iter()
            .map(|(_, size)| *size)
            .collect::<Vec<_>>();

        // find where miss ratio becomes negative, and remove those results, and repective turning points
        if let Some(pos) = new_miss_ratio.iter().position(|&x| x < 0.0) {
            new_miss_ratio.truncate(pos);
            new_turning_points.truncate(pos);
        }

        Self {
            miss_ratio: new_miss_ratio.into_boxed_slice(),
            turning_points: new_turning_points.into_boxed_slice(),
        }
    }

    pub fn new(ri_dist: &[(isize, f64)]) -> Self {
        let mut miss_ratio = ri_dist.iter().map(|(_, prob)| *prob).collect::<Vec<_>>();
        let mut rolling_sum = 1.0 - miss_ratio.iter().sum::<f64>();
        for i in miss_ratio.iter_mut().rev() {
            let current = *i;
            *i = rolling_sum;
            rolling_sum += current;
        }
        let mut turning_points = vec![0.0; miss_ratio.len()];
        let mut prev = 0.0;
        for (i, iter) in turning_points.iter_mut().enumerate().skip(1) {
            *iter = prev + miss_ratio[i - 1] * (ri_dist[i].0 - ri_dist[i - 1].0) as f64;
            prev = *iter;
        }
        let miss_ratio = miss_ratio.into_boxed_slice();
        let turning_points = turning_points.into_boxed_slice();
        Self {
            miss_ratio,
            turning_points,
        }
    }

    #[cfg(feature = "plotters")]
    pub fn plot_miss_ratio_curve<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
    ) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
        let turning_points = &self.turning_points;
        let miss_ratio = &self.miss_ratio;
        area.fill(&WHITE)?;
        let max_x = *turning_points.last().unwrap_or(&1.0);

        let mut chart = ChartBuilder::on(area)
            .caption("Miss Ratio Curve", ("sans-serif", 40))
            .margin(50)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((0.0..max_x).log_scale(), 0.0..1.0)?;

        chart
            .configure_mesh()
            .x_desc("Cache Size")
            .y_desc("Miss Ratio")
            .draw()?;

        let mut points = Vec::new();

        for i in 0..miss_ratio.len() {
            points.push((turning_points[i], miss_ratio[i]));
            if i + 1 < turning_points.len() {
                points.push((turning_points[i + 1], miss_ratio[i]));
            }
        }

        chart
            .draw_series(LineSeries::new(points, &RED))?
            .label("Miss Ratio")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart.configure_series_labels().border_style(BLACK).draw()?;

        Ok(())
    }

    #[cfg(feature = "charming")]
    pub fn plot_interactive_miss_ratio_curve(&self) -> Chart {
        use charming::element::Step;

        let turning_points = &self.turning_points;
        let miss_ratio = &self.miss_ratio;

        //--------------------------------------------------------------------
        // Build coordinates so ECharts can display a true *step* line.
        // Each turning-point is duplicated: first at the old height, then at
        // the new one – this creates the vertical drop.
        //--------------------------------------------------------------------
        let mut coords: Vec<Vec<f64>> = Vec::with_capacity(turning_points.len());
        for i in 0..turning_points.len() {
            let x = turning_points[i];
            let y = miss_ratio[i];
            coords.push(vec![x, y]); // horizontal
        }

        //--------------------------------------------------------------------
        // Compose the chart.
        //--------------------------------------------------------------------
        Chart::new()
            .title(Title::new().text("Miss-ratio curve"))
            .x_axis(
                Axis::new()
                    .name("Cache size")
                    .type_(AxisType::Value) // ← logarithmic axis :contentReference[oaicite:0]{index=0}
                    .min(0.0)
                    .max(Some(*turning_points.last().unwrap()))
                    .scale(true)
                    .log_base(turning_points.last().unwrap().ceil()), // use base-10 ticks; change if you prefer
            )
            .y_axis(
                Axis::new()
                    .name("Miss ratio")
                    .min(0.0)
                    .type_(AxisType::Value)
                    .max(1.0),
            )
            .tooltip(Tooltip::new().trigger(Trigger::Axis)) // hover shows (x, y)
            .series(
                Line::new()
                    .name("Miss ratio")
                    .data(coords) // (x, y) coordinate pairs
                    .step(Step::End) // horizontal → drop at the *end* of an interval
                    .show_symbol(false),
            )
    }
}
