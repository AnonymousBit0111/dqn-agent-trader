use crate::state::State;

pub struct Experience<'a> {
    pub state: State<'a>,
    pub immediate_reward: f32,
    pub action: usize,
    pub next_state: State<'a>,
    pub done: bool,
}
