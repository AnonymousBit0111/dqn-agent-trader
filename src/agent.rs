use std::{collections::VecDeque, error::Error, os::macos::raw::stat};

use crate::{
    data::{Candle, DataPoint},
    device::DEVICE,
    experience::Experience,
    mlp::{self, MultiLayerPerceptron},
    state::State,
};
use candle_core::{backprop::GradStore, DType, Tensor, D};
use candle_nn::{linear, loss, AdamW, Linear, Module, Optimizer, VarBuilder, VarMap};
use rand::{
    rng,
    seq::{IndexedRandom, IteratorRandom},
    Rng,
};

const INPUT_DIM: usize = (7 * 100) + 2;
// --- HELPER FUNCTION FOR MANUAL GRADIENT CLIPPING (CORRECTED) ---
/// Iterates through all gradients in a Grads object and clamps their
/// values within the specified min/max range. This function modifies the
/// Grads object in place.
fn clip_gradients(
    grad_store: &mut GradStore,
    var_map: &VarMap,
    max_norm: f32,
) -> Result<f32, candle_core::Error> {
    let mut total_norm_sq: f32 = 0.0;
    let mut grads = vec![];

    for var in var_map.all_vars() {
        let tensor = var.as_tensor();
        if let Some(grad) = grad_store.get(tensor) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_norm_sq += norm_sq;
            grads.push((var, grad.clone()));
        }
    }

    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for (var, grad) in grads {
            let tensor = var.as_tensor();
            let scale_t = Tensor::new(scale, &tensor.device())?;
            let clipped = grad.broadcast_mul(&scale_t)?;
            grad_store.insert(tensor, clipped);
        }
    }

    Ok(total_norm)
}
pub struct DQNAgent<'a> {
    policy_mlp: MultiLayerPerceptron,
    target_mlp: MultiLayerPerceptron,

    experience_replay_buffer: VecDeque<Experience<'a>>,
    steps_done: usize,
    epsilon_start: f32,
    epsilon_end: f32,
    epsilon_decay: f32,
}

impl<'a> DQNAgent<'a> {
    pub fn new() -> Self {
        let policy = MultiLayerPerceptron::new(&[(7 * 100) + 2, 256, 3]);
        Self {
            target_mlp: policy.clone(),
            policy_mlp: policy,
            experience_replay_buffer: VecDeque::with_capacity(10_000), // input layer is last 100 datapoints + unrealised pnl + is_in_position
            steps_done: 0,
            epsilon_start: 0.9,
            epsilon_end: 0.05,
            epsilon_decay: 100_000.0*20.0,
        }
    }
    pub fn get_current_epsilon(&self) -> f32 {
        self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
                * (-1.0 * self.steps_done as f32 / self.epsilon_decay).exp()
    }

    pub fn reset(&mut self) {
        self.experience_replay_buffer = VecDeque::new();
    }

    pub fn create_input_tensor_vec_from_state(state: &State) -> Vec<f32> {
        let pos: f32 = if state.is_in_position { 1.0 } else { -1.0 };
        let mut input_vec: Vec<f32> = Vec::new();
        input_vec.reserve(INPUT_DIM);
        assert_eq!(state.datapoints.len(), 100);
        for point in state.datapoints.iter() {
            input_vec.extend_from_slice(&[
                point.open,
                point.high,
                point.low,
                point.close,
                point.volume,
                point.range,
                point.real_body_value,
            ]);
        }
        input_vec.extend_from_slice(&[state.unrealised_pnl, pos]);
        input_vec
    }
    pub fn create_input_tensor(
        points: &[DataPoint],
        unrealised_pnl: f32,
        is_in_position: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let pos: f32 = if is_in_position { 1.0 } else { -1.0 };
        let mut input_vec: Vec<f32> = Vec::new();
        input_vec.reserve(7 * points.len() + 2);
        assert_eq!(points.len(), 100);
        for point in points.iter() {
            input_vec.extend_from_slice(&[
                point.open,
                point.high,
                point.low,
                point.close,
                point.volume,
                point.range,
                point.real_body_value,
            ]);
        }
        input_vec.extend_from_slice(&[unrealised_pnl, pos]);
        let len = input_vec.len();
        Tensor::from_vec(input_vec, (1, len), &DEVICE)
    }

    pub fn create_input_tensor_with_state(state: &State) -> Result<Tensor, candle_core::Error> {
        Self::create_input_tensor(state.datapoints, state.unrealised_pnl, state.is_in_position)
    }

    pub fn get_q_values_unmasked(
        &self,
        points: &[DataPoint],
        unrealised_pnl: f32,
        is_in_position: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let input_tensor = Self::create_input_tensor(points, unrealised_pnl, is_in_position)?;
        self.policy_mlp.output(input_tensor)
    }

    pub fn get_q_values(
        &self,
        points: &[DataPoint],
        unrealised_pnl: f32,
        is_in_position: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let raw_output = self.get_q_values_unmasked(points, unrealised_pnl, is_in_position)?;

        let mut mask: Vec<f32> = vec![0.0, 0.0, 0.0];

        if !is_in_position {
            mask[0] = f32::NEG_INFINITY;
        } else {
            mask[2] = f32::NEG_INFINITY;
        }
        let mask_tensor = Tensor::from_vec(mask, (1, 3), &DEVICE)?;
        raw_output + mask_tensor
    }
    pub fn get_decision(
        &self,
        points: &[DataPoint],
        unrealised_pnl: f32,
        is_in_position: bool,
    ) -> Result<usize, candle_core::Error> {
        let mut rng = rng();
        let epsilon_threshold = self.get_current_epsilon();

        if rng.random::<f32>() < epsilon_threshold {
            let mut valid_actions: Vec<usize> = vec![1]; // Action 1 (Hold) is always valid.
            if is_in_position {
                valid_actions.push(0); // If in a position, we can also Sell.
            } else {
                valid_actions.push(2); // If not in a position, we can also Buy.
            }
            return Ok(*valid_actions.choose(&mut rng).unwrap());
        }

        let output = self.get_q_values(points, unrealised_pnl, is_in_position)?;
        let action_tensor = output.argmax(D::Minus1)?;
        let action_index = action_tensor.to_scalar::<u32>()? as usize;

        Ok(action_index)
        // encoding is as follows :
        // 0 -> sell, 1 -> hold , 2 -> buy
    }
    pub fn get_decision_with_state(&self, state: &State) -> Result<usize, candle_core::Error> {
        let mut rng = rng();
        let epsilon_threshold = self.get_current_epsilon();

        if rng.random::<f32>() < epsilon_threshold {
            let mut valid_actions: Vec<usize> = vec![1]; // Action 1 (Hold) is always valid.
            if state.is_in_position {
                valid_actions.push(0); // If in a position, we can also Sell.
            } else {
                valid_actions.push(2); // If not in a position, we can also Buy.
            }
            return Ok(*valid_actions.choose(&mut rng).unwrap());
        }
        let output =
            self.get_q_values(state.datapoints, state.unrealised_pnl, state.is_in_position)?;
        let action_tensor = output.argmax(D::Minus1)?;
        let action_index = action_tensor.get(0)?.to_scalar::<u32>()? as usize;

        Ok(action_index)
        // encoding is as follows :
        // 0 -> sell, 1 -> hold , 2 -> buy
    }

    pub fn add_experience(&mut self, experience: Experience<'a>) {
        if self.experience_replay_buffer.len() >= 10_000 {
            self.experience_replay_buffer.pop_front();
        }
        self.experience_replay_buffer.push_back(experience);
        self.steps_done += 1;
        if (self.steps_done % 10_000 == 0) {
            self.target_mlp = self.policy_mlp.clone();
        }
    }

    pub fn learn(&mut self) -> Result<(), candle_core::Error> {
        if self.experience_replay_buffer.len() < 10_000 || self.steps_done % 10 != 0 {
            return Ok(());
        }

        let mut rng = rng();
        let sample = self
            .experience_replay_buffer
            .iter()
            .choose_multiple(&mut rng, 100);

        let mut states = Vec::with_capacity(100 * INPUT_DIM);
        let mut next_states = Vec::with_capacity(100 * INPUT_DIM);
        let mut rewards = Vec::with_capacity(100);
        let mut dones: Vec<f32> = Vec::with_capacity(100);
        let mut actions = Vec::with_capacity(100);
        for experience in sample {
            states.extend(Self::create_input_tensor_vec_from_state(&experience.state));
            next_states.extend(Self::create_input_tensor_vec_from_state(
                &experience.next_state,
            ));
            rewards.push(experience.immediate_reward);
            dones.push(if experience.done { 1.0 } else { 0.0 });
            actions.push(experience.action as u32);
        }

        let state_tensor = Tensor::from_slice(&states, (100, INPUT_DIM), &DEVICE)?;
        let next_state_tensor = Tensor::from_slice(&next_states, (100, INPUT_DIM), &DEVICE)?;
        let reward_tensor = Tensor::from_slice(&rewards, (100, 1), &DEVICE)?;
        let done_tensor = Tensor::from_slice(&dones, (100, 1), &DEVICE)?;
        let action_tensor = Tensor::from_slice(&actions, (100, 1), &DEVICE)?;

        let predicted_q_values = self.policy_mlp.output(state_tensor)?;
        let next_q_values = self.target_mlp.output(next_state_tensor)?;
        let max_next_q = next_q_values.max(D::Minus1)?.reshape((100, 1))?;
        let expected_q_values = (reward_tensor + (max_next_q * 0.99)? * (1.0 - done_tensor)?)?;
        let relevant_q_values = predicted_q_values.gather(&action_tensor, D::Minus1)?;
        let loss = loss::mse(&relevant_q_values, &expected_q_values)?;

        let mut grads = loss.backward()?;
        let max = clip_gradients(&mut grads, &self.policy_mlp.var_map, 1.0)?;
        self.policy_mlp.optimiser.step(&grads)?;
        self.policy_mlp.clamp_weights(-1e10, 1e10)?;

        Ok(())
    }
}
