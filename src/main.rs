use esp_idf_triton::network::network::Network;


fn main() {
    //Dense Example
    let model_unparsed = "D|3|2|10.845654 11.002682 -13.501029 -14.699452 -53.440483 -53.715294|-6.101849 49.06853 61.28852#D|1|3|30.350481 -78.40228 70.861206|-19.532055#D|1|1|15.161753|-3.7315714".to_string();
    let mut new_net = Network::deserialize_triton_fmt_string(model_unparsed);
    println!("1 and 0: {:?}", new_net.predict(&vec![1.0,0.0])[0]);
    println!("0 and 1: {:?}", new_net.predict(&vec![0.0,1.0])[0]);
    println!("1 and 1: {:?}", new_net.predict(&vec![1.0,1.0])[0]);
    println!("0 and 0: {:?}", new_net.predict(&vec![0.0,0.0])[0]);
}
