use candle_core::Device;
use once_cell::sync::Lazy;

pub static DEVICE: Lazy<Device> = Lazy::new(|| {
    // return Device::Cpu;
    // match Device::new_metal(0) {
    //     Ok(device) => {
    //         println!("Successfully initialized Metal device.");
    //         device
    //     }
    //     Err(err) => {
    //         eprintln!(
    //             "Failed to initialize Metal device: {:?}. Falling back to CPU.",
    //             err
    //         );

    //         Device::Cpu
    //     }
    // }
    // You could also try CUDA here if targeting multiple GPU backends:
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("Successfully initialized CUDA device.");
            device
        }
        Err(err_cuda) => {
            eprintln!(
                "Failed to initialize CUDA device: {:?}. Falling back to CPU.",
                err_cuda
            );
            Device::Cpu
        }
    }
});
