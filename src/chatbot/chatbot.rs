use anyhow::Result;
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::model::{Decoder, Encoder, Vocab};
use crate::chatbot::generation::generate_text;

pub fn run_chatbot(
    model_path: &PathBuf,
    vocab_path: &PathBuf,
    _data_path: &PathBuf,
    dim: usize,
    max_ctx: usize,
    num_layers: usize,
    num_heads: usize,
) -> Result<()> {
    println!("Loading transformer chatbot model...");
    
    let device = match Device::new_cuda(0) {
        Ok(device) => {
            println!("Using device: CUDA(0)");
            device
        }
        Err(_) => {
            println!("Using device: CPU");
            Device::Cpu
        }
    };
    
    // Load vocab
    let vocab_text = fs::read_to_string(vocab_path)?;
    let mut vocab = Vocab::new();
    for line in vocab_text.lines() {
        if !line.is_empty() {
            vocab.add_token(line);
        }
    }
    println!("Loaded vocab: {} tokens", vocab.id_to_token.len());
    
    // Load model weights
    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create encoder and decoder with same architecture as training
    let encoder = Encoder::new(
        vb.pp("encoder"),
        vocab.id_to_token.len(),
        dim,
        num_layers,
        num_heads,
    )?;
    
    let decoder = Decoder::new(
        vb.pp("decoder"),
        dim,
        vocab.id_to_token.len(),
        num_layers,
        num_heads,
    )?;
    
    println!("Model loaded! (dim={}, layers={}, heads={})", dim, num_layers, num_heads);
    println!("\nReady to chat! (type 'quit' or 'exit' to end)\n");
    
    // Chat loop
    loop {
        print!("You: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() || input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        // Generate response
        match generate_text(
            &encoder,
            &decoder,
            &vocab,
            input,
            max_ctx,
            32, // max generated tokens
            &device,
        ) {
            Ok(response) => {
                if response.is_empty() {
                    println!("Bot: ...\n");
                } else {
                    println!("Bot: {}\n", response);
                }
            }
            Err(e) => {
                eprintln!("Generation error: {}", e);
                println!("Bot: (error)\n");
            }
        }
    }
    
    Ok(())
}
