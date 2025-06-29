use std::error::Error;

use candle_core::{DType, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, VarBuilder, VarMap, linear};

use crate::device::DEVICE;

pub struct MultiLayerPerceptron {
    pub layers: Vec<Linear>,
    pub var_map: VarMap,
    pub topology: Vec<usize>,
    pub optimiser: AdamW,
}

impl MultiLayerPerceptron {
    pub fn new(topology: &[usize]) -> Self {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &DEVICE);
        let mut layers: Vec<Linear> = Vec::with_capacity(topology.len() - 1);

        for i in 0..(topology.len() - 1) {
            let layer =
                linear(topology[i], topology[i + 1], vb.pp(i)).expect("Failed to create layer");
            layers.push(layer);
        }

        let adam = candle_nn::AdamW::new(
            vm.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 1e-4,
                weight_decay: 0.01,
                ..Default::default()
            },
        )
        .expect("Failed to create AdamW optimizer");

        Self {
            layers,
            topology: topology.to_vec(),
            var_map: vm,
            optimiser: adam,
        }
    }
    pub fn clamp_weights(&self, min: f32, max: f32) -> Result<(), candle_core::Error> {
        // Lock the VarMap to get access to the named tensors.
        let vars = self.var_map.data().lock().unwrap();

        // Iterate through all the variables (weights and biases).
        for var_tensor in vars.values() {
            // Perform the in-place clamp operation on the tensor data.
            // This will replace any value < min with min, and any value > max with max.
            let clamped_tensor = var_tensor.clamp(min, max)?;
            var_tensor.set(&clamped_tensor)?;
        }
        Ok(())
    }

    pub fn output(&self, inputs: Tensor) -> Result<Tensor, candle_core::Error> {
        let mut x = inputs;
        for layer in self.layers.iter().take(self.layers.len() - 1) {
            x = layer.forward(&x)?;
            x = x.relu()?;
        }
        if let Some(last_layer) = self.layers.last() {
            x = last_layer.forward(&x)?;
        }
        Ok(x)
    }
}

impl Clone for MultiLayerPerceptron {
    fn clone(&self) -> Self {
        // 1. Create a brand new, independent MLP.
        //    This new MLP has its own VarMap and its own random weights.
        let new_mlp = MultiLayerPerceptron::new(&self.topology);

        // --- SCOPE ADDED TO FIX BORROW CHECKER ERROR ---
        // By creating a new scope, the MutexGuards (`source_vars` and `new_vars`)
        // are dropped at the end of this block, releasing their locks.
        {
            // 2. Get all the named tensors (weights and biases) from the *source* network.
            let source_vars = self.var_map.data().lock().unwrap();

            // 3. Get all the named tensors from the *new* network we just created.
            let new_vars = new_mlp.var_map.data().lock().unwrap();

            // 4. Iterate through the source variables and copy their data into the new variables.
            for (name, source_var) in source_vars.iter() {
                if let Some(target_var) = new_vars.get(name) {
                    // .set() performs a deep copy of the tensor data.
                    target_var
                        .set(source_var.as_tensor())
                        .expect("Failed to copy tensor data");
                }
            }
        } // --- LOCKS ARE RELEASED HERE ---

        // 5. Return the new MLP. The move is now safe as there are no active borrows.
        new_mlp
    }
}
