#[cfg(feature = "charming")]
use charming::{
    Chart,
    component::{Axis, Title},
    element::{AxisType, Tooltip, Trigger},
    series::Line,
};
use core::f64;
#[cfg(feature = "plotters")]
use plotters::{coord::Shift, prelude::*};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Serialize, Deserialize)]
pub struct MissRatioCurve {
    miss_ratio: Box<[f64]>,
    turning_points: Box<[f64]>,
}

impl MissRatioCurve {
    fn binomial(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }

        let mut result = 1.0;
        for i in 0..k {
            result = result * (n - i) as f64 / (i + 1) as f64;
        }
        result
    }

    pub fn compute_assoc(&self, associativity: usize) -> Self {
        let len = self.turning_points.len();
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

        let max_tp_usize = self.turning_points[len - 1].ceil() as usize;
        let cache_sizes: Vec<_> = (associativity..(max_tp_usize + associativity))
            .step_by(associativity)
            .collect();

        // Calculate all miss ratios in parallel and show progress
        let pb = ProgressBar::new(cache_sizes.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .tick_chars("◐◓◑◒ "),
        );

        let parallel_results: Vec<(f64, f64)> = cache_sizes
            .par_iter()
            .map_with(pb.clone(), |pb, &cache_size| {
                let mut miss_ratio = 1.0;
                let number_of_sets = (cache_size / associativity) as f64;
                for i in 1..associativity + 1 {
                    for j in 0..len {
                        if self.turning_points[j].ceil() as usize >= i {
                            miss_ratio -= rd_portions[j]
                                * (1.0 / number_of_sets).powi(i as i32 - 1)
                                * ((number_of_sets - 1.0) / number_of_sets)
                                    .powi((self.turning_points[j].ceil() as usize - i) as i32)
                                * MissRatioCurve::binomial(
                                    self.turning_points[j].ceil() as usize - 1,
                                    i,
                                );
                        }
                    }
                }
                pb.inc(1);
                (miss_ratio, cache_size as f64)
            })
            .collect();

        pb.finish_and_clear();

        // Create the final vectors with initial values
        let mut new_miss_ratio = vec![1.0];
        let mut new_turning_points = vec![0.0];

        // Extend with parallel results
        new_miss_ratio.extend(parallel_results.iter().map(|(ratio, _)| *ratio));
        new_turning_points.extend(parallel_results.iter().map(|(_, size)| *size));

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
