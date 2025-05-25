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
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MissRatioCurve {
    miss_ratio: Box<[f64]>,
    turning_points: Box<[f64]>,
}

impl MissRatioCurve {
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
