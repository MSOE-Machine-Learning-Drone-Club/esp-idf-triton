 <img align="left" src="https://raw.githubusercontent.com/BradenEverson/triton/master/triton-logo.svg" width="80px" height="80px" alt="triton mascot icon">

# esp-idf-triton

### Lightweight fork of the [Triton](https://github.com/BradenEverson/triton) crate meant for running precompiled models on embedded systems

[![crates.io](https://img.shields.io/crates/v/esp_idf_triton.svg)](https://crates.io/crates/esp_idf_triton)
[![Documentation](https://docs.rs/esp_idf_triton/badge.svg)](https://docs.rs/esp_idf_triton)

esp-idf-triton can be used to run precompiled ML models created by the [main triton](https://github.com/BradenEverson/triton) crate. Creating a model in a Triton program and using the ```network.serialize_triton_fmt("out.triton");``` will create a .triton file that is compatible with being deserialized by esp-idf-triton.

File IO support is soon to come, but as a quick proof of concept esp-idf-triton can currently parse neural network strings copied from the .triton file and perform forwards predictions.

## Example

```rust
let model_str = "D|3|2|10.845654 11.002682 -13.501029 -14.699452 -53.440483 -53.715294|
    -6.101849 49.06853 61.28852#D|1|3|30.350481 -78.40228 70.861206|-19.532055#D|1|1|15.161753|-3.7315714".to_string();

let mut xor_net = Network::deserialize_triton_fmt_string(model_str);
```
creates a simple Network based on a trained SIGMOID XoR predictor.

The .triton file format currently defaults to SIGMOID activations, but its a top priority to implement activation function support in this file format

#### If open source development is your thing, we at Triton would love additional work on anything that can be implemented, please contact **eversonb@msoe.edu** if you'd like to help out!

# License
Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
