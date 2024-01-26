use super::layer::layers::Layer;
use super::input::Input;
use super::serialize::ser_layer::SerializedLayer;

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network{
    pub fn new() -> Network {
        Network { layers: vec![] }
    }
    pub fn predict(&mut self, input: &dyn Input) -> Vec<f32>{
        let in_box: Box<dyn Input> = input.to_box();
        self.feed_forward(&in_box)
    }
    ///Travels through a neural network's abstracted Layers and returns the resultant vector at the
    ///end
    ///
    ///# Arguments
    ///* `input_obj` - Any structure that implements the Input trait to act as an input to the data
    ///# Returns
    ///A vector at the end of the feed forward
    ///
    ///# Examples
    ///
    ///```
    ///let new_net = Network::New();
    ///new_new.add_layer(LayerTypes::Dense(2, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(3, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(4, Activations::SIGMOID, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(2, Activations::TANH, 0.01));
    ///new_new.add_layer(LayerTypes::Dense(1, Activations::SIGMOID, 0.01));
    ///
    ///new_net.compile()
    ///
    ///let res = new_net.feed_forward(vec![1.0, 0.54]);
    ///```
    fn feed_forward(&mut self, input_obj: &Box<dyn Input>) -> Vec<f32> {
        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
        }
        data_at.to_param().to_owned()
    }
    pub fn deserialize_triton_fmt_string(format_string: String) -> Network {
        let mut net: Network = Network::new();
        let parse_triton = format_string.split("#");
        for layer in parse_triton {
            let new_layer: Box<dyn Layer> = SerializedLayer::from_string(layer.to_string()).from();
            net.layers.push(new_layer);
        }
        net
    }
}
