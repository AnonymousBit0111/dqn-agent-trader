pub struct DataPoint {
    pub open: f32,
    pub close: f32,
    pub volume: f32,
    pub high: f32,
    pub low: f32,
    pub real_body_value: f32,
    pub range: f32,
}

#[derive(Clone)]

pub struct Candle {
    pub open: f32,
    pub close: f32,
    pub volume: f32,
    pub high: f32,
    pub low: f32,
}
