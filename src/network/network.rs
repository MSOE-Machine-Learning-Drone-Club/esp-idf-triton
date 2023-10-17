use super::{matrix::Matrix, activations::Activation};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>
}

impl Network<'_>{
    pub fn new<'a>(layers: Vec<usize>, activation: Activation<'a>, learning_rate: f64) -> Network{
        let mut net = Network{
            layers: layers,
            weights: vec![],
            biases: vec![],
            data: vec![],
            learning_rate: learning_rate,
            activation
        };

        for i in 0..net.layers.len() - 1 {
            net.weights.push(Matrix::new_random(net.layers[i+1],net.layers[i]));
            net.biases.push(Matrix::new_random(net.layers[i+1], 1));
        }
        net
    }
    pub fn feed_to_point(&mut self, inputs: &Vec<f64>, layer_to: usize) -> Vec<f64>{
         if inputs.len() != self.layers[0]{
             panic!("Invalid numer of inputs");
         }
         if layer_to >= self.layers.len(){
             panic!("To destination is larger than network size");
         }
         let mut current = Matrix::from(vec![inputs.clone()]).transpose();
         self.data = vec![current.clone()];

         for i in 0..layer_to{
             current = ((self.weights[i].clone()
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.function);
             self.data.push(current.clone());
         }
         current.transpose().data[0].to_owned()
    }
    pub fn point_to_feed(&mut self, inputs_at_point: &Vec<f64>, layer_at: usize) -> Vec<f64> {
        if inputs_at_point.len() != self.layers[layer_at]{
            panic!("Invalid number of inputs for layer");
        }
        let mut current = Matrix::from(vec![inputs_at_point.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = ((self.weights[i].clone()
                        * &current)
                        + &self.biases[i])
                        .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }
         
        let mut current = Matrix::from(vec![inputs.clone()]).transpose();
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len()-1{
            current = ((self.weights[i].clone()
                 * &current)
                 + &self.biases[i]) 
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn back_propegate(&mut self, outputs: Vec<f64>, targets: Vec<f64>){
        if targets.len() != self.layers[self.layers.len()-1] {
            panic!("Invalid number of targets found :(");
        }
        let mut parsed = Matrix::from(vec![outputs]).transpose();
        //println!("{} {}",parsed.rows, parsed.columns);
        
        let mut errors = Matrix::from(vec![targets]).transpose() - &parsed;

        let mut gradients = parsed.map(self.activation.derivative);
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].clone() + &(gradients.clone() * (&self.data[i].transpose()));

            self.biases[i] = self.biases[i].clone() + &gradients;
            errors = self.weights[i].transpose() * (&errors);

            gradients = self.data[i].map(self.activation.derivative);
        }
    }
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for i in 1..=epochs{
            if epochs < 100 || i % (epochs/100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len(){
                let outputs = self.feed_forward(&inputs[j]);
                self.back_propegate(outputs, targets[j].clone());
            }
        }
    }
}