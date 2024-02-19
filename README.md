  <img align="left" src="https://raw.githubusercontent.com/BradenEverson/unda/master/unda.svg" width="75px" height="75px" alt="unda icon">

# esp-idf-unda

### Lightweight fork of the [unda](https://github.com/BradenEverson/unda) crate meant for running precompiled models on embedded systems

[![crates.io](https://img.shields.io/crates/v/esp_idf_unda.svg)](https://crates.io/crates/esp_idf_unda)
[![Documentation](https://docs.rs/esp_idf_unda/badge.svg)](https://docs.rs/esp_idf_unda)

esp-idf-unda can be used to run precompiled ML models created by the [unda](https://github.com/BradenEverson/unda) crate. Creating a model in a Unda program and using the ```network.serialize_unda_fmt("out.unda");``` will create a .unda file that is compatible with being deserialized by esp-idf-unda.

File IO support is soon to come, but as a quick proof of concept esp-idf-unda can currently parse neural network strings copied from the .unda file and perform forwards predictions.

## Example

```rust
let model_str = "D|3|2|10.845654 11.002682 -13.501029 -14.699452 -53.440483 -53.715294|
    -6.101849 49.06853 61.28852#D|1|3|30.350481 -78.40228 70.861206|-19.532055#D|1|1|15.161753|-3.7315714".to_string();

let mut xor_net = Network::deserialize_unda_fmt_string(model_str);
```
creates a simple Network based on a trained SIGMOID XoR predictor.

The .unda file format currently defaults to SIGMOID activations, but its a top priority to implement activation function support in this file format

#### If open source development is your thing, we at unda would love additional work on anything that can be implemented, please contact **eversonb@msoe.edu** if you'd like to help out!

# License
Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
