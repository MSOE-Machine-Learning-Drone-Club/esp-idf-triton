use crate::network::{input::Input, matrix::Matrix};


pub trait Layer: Send + Sync{
    ///Propegates the data forward through 1 layer and returns
    ///the data created from that respective layer as a dynamic Input object
    ///
    ///Ex: A Dense layer will take in a 1 dimensional vector and return out a 1 dimensional vector
    ///A convolutional layer will return a 3 dimensional matrix of all filter applications
    ///
    ///Both of these types implement the Input trait
    fn forward(&self, _inputs: &Box<dyn Input>) -> Box<dyn Input> {
        Box::new(Matrix::new_empty(0,0))
    }
}
