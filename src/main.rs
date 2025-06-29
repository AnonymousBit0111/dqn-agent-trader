use std::{error::Error, fs::File, time::Instant};

use polars::{
    error::PolarsError,
    frame::DataFrame,
    io::SerReader,
    prelude::{CsvReader, IntoLazy, LazyFrame, col},
};

use crate::{
    data::{Candle, DataPoint},
    environment::Environment,
};

mod agent;
mod data;
mod device;
mod environment;
mod experience;
mod mlp;
mod state;

fn get_candle_data(frame: &DataFrame) -> Result<Vec<Candle>, PolarsError> {
    let open: Vec<f32> = frame
        .column("Open")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let close: Vec<f32> = frame
        .column("Close")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let volume: Vec<f32> = frame
        .column("Volume")?
        .i64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let high: Vec<f32> = frame
        .column("High")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let low: Vec<f32> = frame
        .column("Low")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let data: Vec<Candle> = open
        .iter()
        .enumerate()
        .map(|(i, val)| Candle {
            open: *val,
            close: close[i],
            volume: volume[i],
            high: high[i],
            low: low[i],
        })
        .collect();

    Ok(data)
}
fn get_tensor_data(frame: &DataFrame) -> Result<Vec<DataPoint>, PolarsError> {
    let open_norm: Vec<f32> = frame
        .column("Open Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let close_norm: Vec<f32> = frame
        .column("Close Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let volume_norm: Vec<f32> = frame
        .column("Volume Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let high_norm: Vec<f32> = frame
        .column("High Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let low_norm: Vec<f32> = frame
        .column("Low Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let real_body_value_norm: Vec<f32> = frame
        .column("Real Body Value Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();
    let range_norm: Vec<f32> = frame
        .column("Range Norm")?
        .f64()?
        .to_vec()
        .iter()
        .map(|opt| opt.unwrap() as f32)
        .collect();

    let data: Vec<DataPoint> = open_norm
        .iter()
        .enumerate()
        .map(|(i, val)| DataPoint {
            open: *val,
            close: close_norm[i],
            volume: volume_norm[i],
            high: high_norm[i],
            low: low_norm[i],
            real_body_value: real_body_value_norm[i],
            range: range_norm[i],
        })
        .collect();

    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("data/data.csv")?;
    let mut df = CsvReader::new(file).finish()?;
    // --- DATA SANITIZATION STEP ---
    // Get all column names to iterate through them.
    let column_names: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    for name in column_names {
        // We only want to fill NaNs in numeric columns.
        if df.column(&name)?.dtype().is_primitive_numeric() {
            df = df
                .lazy()
                .with_column(
                    // For the current column, fill any NaN values with 0.0.
                    // You can also use other strategies like filling with the mean/median.
                    col(&name).fill_nan(0),
                )
                .collect()?;
        }
    }

    let start = Instant::now();
    let datapoints = get_tensor_data(&df)?;
    let candles = get_candle_data(&df)?;
    println!("took {}ms to parse data", start.elapsed().as_millis());

    let mut env = Environment::new();
    for epoch in 0.. {
        env.run_episode(&candles, &datapoints, epoch)?;
    }
    Ok(())
}
