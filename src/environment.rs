use std::fs::File;

use candle_nn::VarMap;
use polars::{
    frame::DataFrame,
    io::SerWriter,
    prelude::{Column, CsvWriter, LazyFrame, NamedFrom},
};

use crate::{
    agent::DQNAgent,
    data::{self, Candle, DataPoint},
    experience::Experience,
    state::State,
};

const TRANSACTION_COST_PERCENT: f32 = 0.1 / 100.0; // 0.1%

pub struct Environment<'a> {
    dqn_agent: DQNAgent<'a>,
}

impl<'a> Environment<'a> {
    pub fn new() -> Self {
        Self {
            dqn_agent: DQNAgent::new(),
        }
    }

    pub fn run_episode(
        &mut self,
        candles: &[Candle],
        datapoints: &'a [DataPoint],
        epoch: i32,
    ) -> Result<(), candle_core::Error> {
        // --- METRICS INITIALIZATION (ADDED) ---
        let mut total_pnl: f32 = 0.0;
        let mut number_of_trades = 0;
        // ---

        let mut steps_log: Vec<u32> = Vec::new();
        let mut price_log: Vec<f32> = Vec::new();
        let mut decision_log: Vec<u32> = Vec::new();
        let mut unrealised_pnl_log: Vec<f32> = Vec::new();
        let mut total_pnl_log: Vec<f32> = Vec::new();
        // ---

        let mut unrealised_pnl: f32 = 0.0;
        let mut is_in_position = false;
        let mut last_buy: Option<&Candle> = None;

        self.dqn_agent.reset();
        for i in 100..candles.len() - 1 {
            let points = &datapoints[i - 100..i];
            let candle = &candles[i];

            if last_buy.is_some() {
                unrealised_pnl = (candle.close - last_buy.unwrap().open) / last_buy.unwrap().open;
                is_in_position = true;
            } else {
                unrealised_pnl = 0.0;
            }
            let state = State {
                candle_index: i,
                unrealised_pnl,
                is_in_position,
                datapoints: points,
            };

            let decision = self.dqn_agent.get_decision_with_state(&state)?;

            // --- LOGGING THE DECISION (ADDED) ---
            let action_str = match decision {
                0 => "Sell",
                1 => "Hold",
                2 => "Buy",
                _ => "Unknown",
            };
            if decision != 1 {
                println!(
                    "[Step {}] In Position: {}. PnL: {:.9}. Decision: {}",
                    i, is_in_position, unrealised_pnl, action_str
                );
            }

            // ---

            match decision {
                0 => {
                    // sell
                    // --- METRICS UPDATE (ADDED) ---
                    if is_in_position {
                        // --- APPLY TRANSACTION COSTS HERE ---
                        // A round trip involves two transactions: the initial buy and this sell.
                        let total_cost_percentage = 2.0 * TRANSACTION_COST_PERCENT;

                        // The 'unrealised_pnl' is the gross profit percentage before costs.
                        let pnl_after_costs = unrealised_pnl - total_cost_percentage;

                        println!(
                            "    -> Trade Closed. Gross PnL: {:.4}%. Net PnL after costs: {:.4}% (not included in log)",
                            unrealised_pnl * 100.0, // Display as a percentage
                            pnl_after_costs * 100.0
                        );

                        // Add the NET PnL to your total.
                        total_pnl += unrealised_pnl;
                        is_in_position = false;
                        last_buy = None;
                        number_of_trades += 1;
                    }
                }
                1 => {
                    // hold
                }
                2 => {
                    // --- METRICS UPDATE (ADDED) ---
                    if !is_in_position {
                        // Ensure we only count a new trade
                        println!("    -> Entered Position at: {:.2}", candle.close);
                        number_of_trades += 1;
                    }
                    // ---
                    last_buy = Some(&candles[i]);
                    is_in_position = true;
                    //buy
                }
                _ => {}
            };

            steps_log.push(i as u32);
            price_log.push(candle.close);
            decision_log.push(decision as u32);
            unrealised_pnl_log.push(unrealised_pnl);
            total_pnl_log.push(total_pnl);
            let transaction_cost = if decision == 0 || decision == 2 {
                // If the agent decided to BUY or SELL, it incurs a cost for this step.
                TRANSACTION_COST_PERCENT * candles[i + 1].close
            } else {
                // If the agent chose to HOLD, there is no transaction cost for this step.
                0.0
            };

            let immediate_reward = candles[i + 1].close - candles[i].close;

            let next_unrealised_pnl = if last_buy.is_some() {
                (candles[i + 1].close - last_buy.unwrap().open) / last_buy.unwrap().open
            } else {
                0.0
            };
            let next_state = State {
                datapoints: &datapoints[i - 99..i + 1],
                candle_index: i + 1,
                unrealised_pnl: next_unrealised_pnl,
                is_in_position,
            };

            let experience = Experience {
                state,
                immediate_reward,
                action: decision,
                next_state,
                done: (i == candles.len() - 2),
            };
            self.dqn_agent.add_experience(experience);
            self.dqn_agent.learn()?;
        }

        // --- FINAL REPORT (ADDED) ---
        println!("\n--- Episode Finished ---");
        println!("Total Realized PnL: {:.2}", total_pnl);
        println!("Total Trades Made: {}", number_of_trades);
        println!("----------------------\n");
        // ---

        println!("Writing metrics to dry_run_output.csv...");
        let mut df = DataFrame::new(vec![
            Column::new("step".into(), steps_log),
            Column::new("price".into(), price_log),
            Column::new("decision".into(), decision_log),
            Column::new("unrealised_pnl".into(), unrealised_pnl_log),
            Column::new("total_pnl".into(), total_pnl_log),
        ])
        .expect("Failed to create DataFrame");

        let mut file = File::create(format!("output/run_output{}.csv", epoch + 1))
            .expect("could not create file");
        CsvWriter::new(&mut file)
            .finish(&mut df)
            .expect("Failed to write DataFrame to CSV");
        println!("...Done.");

        Ok(())
    }
}
