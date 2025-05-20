use anyhow::Result;
use core::f64;
use plotters::prelude::*;
use std::path::Path;

pub struct MissRatioCurve {
    miss_ratio: Vec<f64>,
    turning_points: Vec<f64>,
}

impl MissRatioCurve {
    pub fn new(ri_dist: &[(isize, f64)]) -> Self {
        let mut miss_ratio = ri_dist.iter().map(|(_, prob)| *prob).collect::<Vec<_>>();
        let mut rolling_sum = 0.0;
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
        //trace!("Turning points: {:?}", turning_points);
        //trace!("Miss ratio: {:?}", miss_ratio);
        Self {
            miss_ratio,
            turning_points,
        }
    }
    pub fn plot_miss_ratio_curve<P: AsRef<Path>>(
        &self,
        path: P,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let turning_points = &self.turning_points;
        let miss_ratio = &self.miss_ratio;
        let root_area = SVGBackend::new(&path, (width, height)).into_drawing_area();

        root_area.fill(&WHITE)?;

        let max_x = *turning_points.last().unwrap_or(&1.0);

        let mut chart = ChartBuilder::on(&root_area)
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
}
