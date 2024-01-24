use esp_idf_triton::network::network::Network;


fn main() {
    //Dense Example
    let mut new_net: Network = Network::load("test.json");

    println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(&vec![0.0,0.0])[0]);
}
