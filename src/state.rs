use crate::data::{Candle, DataPoint};

pub struct State<'a> {
    pub candle_index: usize,
    pub unrealised_pnl: f32,
    pub is_in_position: bool,
    pub datapoints: &'a [DataPoint],
}
