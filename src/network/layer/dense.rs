use crate::network::{matrix::Matrix, activations::Activations, input::Input};

use super::layers::Layer;

///A Dense Neural Network Layer of a model, containing just nodes, weights, biases and an
///activation function
///Implements the Layer trait
pub struct Dense{
    pub weights: Matrix,   
    pub biases: Matrix,
    pub data: Matrix,
    loss: f32,

    pub activation_fn: Activations,
    learning_rate: f32,

    beta1: f32,
    beta2: f32,
    epsilon: f32,
    time: usize,

    m_weights: Matrix,
    v_weights: Matrix,
    m_biases: Matrix,
    v_biases: Matrix
}

impl Dense{
    pub fn new_ser(rows: usize, cols: usize, flat_weight: Vec<f32>, flat_bias: Vec<f32>) -> Dense {
        let weight_shape: Matrix = Matrix::from_sized(flat_weight, rows, cols);
        let bias_shape: Matrix = Matrix::from_sized(flat_bias, rows, 1);

        Dense {
            weights: weight_shape,
            biases: bias_shape,
            data: Matrix::new_empty(0, 0),
            loss: 1.0,
            activation_fn: Activations::SIGMOID,
            learning_rate: 0.01,
            beta1: 0.99,
            beta2: 0.99,
            epsilon: 1e-16,
            time: 1,
            m_weights: Matrix::new_empty(0, 0),
            v_weights: Matrix::new_empty(0, 0),
            m_biases: Matrix::new_empty(0, 0),
            v_biases: Matrix::new_empty(0, 0)
        }
    }
}


impl Layer for Dense{
    ///Moves the DNN forward through the weights and biases of this current layer
    ///Maps an activation function and then returns the resultant Matrix
    fn forward(&self, inputs: &Box<dyn Input>) -> Box<dyn Input> {
        let new_data = self.activation_fn.apply_fn(self.weights.clone() * &Matrix::from(inputs.to_param().to_param_2d()).transpose() + &self.biases);

        Box::new(new_data)
    }
}

