use crate::network::{matrix::Matrix,activations::Activations, input::Input, matrix3d::Matrix3D};
use super::layers::Layer;

pub struct Convolutional{
    filter_weights: Matrix3D,
    filter_biases: Vec<f32>,
    data: Matrix3D,
    stride: usize,
    filters: usize,
    shape: (usize, usize),
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    loss: f32,
    
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    time: usize,

    m_weights: Matrix3D,
    v_weights: Matrix3D,
    m_biases: Vec<f32>,
    v_biases: Vec<f32>,

    activation_fn: Activations,
    learning_rate: f32
}

impl Convolutional{
    fn get_betas(&self) -> (f32, f32) {
        (0.9, 0.999)
    }
    fn get_epsilon(&self) -> f32{
        1e-10
    }
    pub fn convolute(&self, idx: usize, input: Matrix) -> Matrix {
        let kernel = self.filter_weights.get_slice(idx);
        let mut output = Matrix::new_empty(self.output_shape.0, self.output_shape.1);

        let mut x: usize;
        let mut y: usize = 0;

        for output_x in 0..output.columns {
            x = 0;
            for output_y in 0..output.rows {
                let sum = input.get_sub_matrix(x, y, kernel.rows, kernel.columns).dot_multiply(&kernel).sum();
                output.data[output_y][output_x] = sum;

                x += self.stride;
            }
            y += 1;
        }

        //println!("{}", output);
        //self.data.set_slice(idx, output);
        output
    }
    fn get_res_size(w: usize, k: usize, p: usize, s:usize) -> usize {
        (w - k + 2*p) / s + 1
    }
}

impl Layer for Convolutional {
    fn forward(&self,inputs: &Box<dyn Input>) -> Box<dyn Input> {
        let input_mat = Matrix3D::from(inputs.to_param_3d());
        for i in 0..input_mat.layers {
            for j in 0..self.filters {
                self.convolute(j, input_mat.get_slice(i));
            }
        }
        let data = self.data.clone() + &self.filter_biases;
        Box::new(data.clone())
    }
}
