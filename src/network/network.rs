use super::layer::distributions::Distributions;
use super::layer::layers::{Layer, LayerTypes};
use super::layer::dense::Dense;
use super::layer::noise::gen_noise;
use super::layer::pair::GradientPair;
use super::matrix::Matrix;
use super::input::Input;
use rand::{RngCore, Rng, thread_rng};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Serialize, Deserialize};

use futures::stream::{StreamExt, FuturesUnordered};

use serde_json::{to_string, from_str};
use std::io;
use std::ops::Range;
use std::{
    fs::File,
    io::{Read,Write},
};

#[derive(Serialize, Deserialize)]
pub struct Network {
    batch_size: usize,
    pub layer_sizes: Vec<usize>,
    pub loss: f32,
    loss_train: Vec<f32>,
    pub layers: Vec<Box<dyn Layer>>,
    uncompiled_layers: Vec<LayerTypes>,
    seed: Option<String>,
    #[serde(skip)]
    #[serde(default = "Network::thread_rng")]
    rng: Box<dyn RngCore>
}

impl Network{
    fn thread_rng() -> Box<dyn RngCore> {
        Box::new(thread_rng())
    }
    pub fn predict(&mut self, input: &dyn Input) -> Vec<f32>{
        let in_box: Box<dyn Input> = input.to_box();
        self.feed_forward(&in_box)
    }
    pub fn set_seed(&mut self, seed: &str){
        self.seed = Some(String::from(seed));
        self.rng = self.get_rng();
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
        if input_obj.to_param().shape() != self.layers[0].shape(){
            panic!("Input shape does not match input layer shape \nInput: {:?}\nInput Layer:{:?}", input_obj.shape(), self.layers[0].shape());
        }
        
        let mut data_at: Box<dyn Input> = Box::new(input_obj.to_param());
        for i in 0..self.layers.len(){
            data_at = self.layers[i].forward(&data_at);
            self.layers[i].set_data(&data_at);
        }
        data_at.to_param().to_owned()
    }
    fn get_rng(&self) -> Box<dyn RngCore> {
        match &self.seed {
            Some(seed_rng) => Box::new(Seeder::from(seed_rng).make_rng::<Pcg64>()),
            None => Box::new(rand::thread_rng())
        }
    }
    pub fn load(path: &str) -> Network{
        let mut buffer = String::new();
        let mut file = File::open(path).expect("Unable to read file :(");

        file.read_to_string(&mut buffer).expect("Unable to read file but even sadder :(");

        let mut net: Network = from_str(&buffer).expect("Json was not formatted well >:(");
        net.rng = net.get_rng();
        net
    }
    pub fn load_str(json_string: &str) -> Network {
        let mut net: Network = from_str(json_string).expect("Improper json format");
        net.rng = net.get_rng();
        net
    }
    pub fn load_cbor(path: &str) -> Result<Network, serde_cbor::Error> {
        let file = File::open(path).expect("error loading file");
        let mut network: Network = serde_cbor::from_reader(file)?;
        network.rng = network.get_rng();
        Ok(network)
    }
    pub fn to_vec(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }
    pub fn from_vec(data: Vec<u8>) -> Result<Network, serde_cbor::Error> {
        serde_cbor::from_slice(&data[..])
    }
}
