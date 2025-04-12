import gradio as gr
import threading
import time
from process import generate
import os
import random
import argparse

def run_generation(genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs):
    try:
        # Generate a random seed if seed is 0
        if seed == 0:
            seed = random.randint(1, 2**31 - 1)  # Use a wide range of positive integers
            print(f"Generated random seed: {seed}")
        
        output_path = generate(genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs)
        return "Generation complete!", output_path
    except Exception as e:
        return f"Error: {str(e)}", None

# Load default content from files
def load_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""

# Get the absolute path to the files
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
genre_path = os.path.join(base_dir, "prompt_egs", "genre.txt")
lyrics_path = os.path.join(base_dir, "prompt_egs", "lyrics.txt")

# Get default values from prompt_egs files
genre_default = load_text_file(genre_path)
lyrics_default = load_text_file(lyrics_path)

# Create the Gradio interface
with gr.Blocks() as demo:
    # Title area
    gr.Markdown("# YuE Gradio GUI (based on YuEGP v3's Gradio GUI)")
    gr.Markdown("""
    Official Website: [YuE](https://github.com/multimodal-art-projection/YuE)
                
    GPU Poor version by DeepBeepMeep ([GitHub](https://github.com/deepbeepmeep/YuEGP)). Switch to profile 1 for fast generation (requires a 16 GB VRAM GPU), 1 min of song will take only 4 minutes
    """)
    
    # Two-column layout
    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            genre_prompt = gr.Textbox(
                label="Genres Prompt (one Genres Prompt per line for multiple generations)",
                placeholder="inspiring female uplifting pop airy vocal electronic bright vocal",
                lines=3,
                value=genre_default
            )
            
            lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="[verse]\nStaring at the sunset, colors paint the sky\nThoughts of you keep swirling, can't deny\nI know I let you down, I made mistakes\nBut I'm here to mend the heart I didn't break\n\n[chorus]\nEvery road you take, I'll be one step behind\nEvery dream you chase, I'm reaching for the light",
                lines=30,
                value=lyrics_default
            )
            
            num_songs = gr.Slider(
                minimum=1, 
                maximum=25, 
                value=1, 
                step=1,
                label="Number of Generated Songs per Genres Prompt"
            )
        
        # Right column
        with gr.Column(scale=1):
            num_sequences = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=2, 
                step=1,
                label="Number of Sequences (paragraphs in Lyrics, the higher this number, the higher the VRAM consumption)"
            )
            
            num_tokens = gr.Slider(
                minimum=300, 
                maximum=6000, 
                value=3000, 
                step=100,
                label="Number of tokens per sequence (1000 tokens = 10s, the higher this number, the higher the VRAM consumption)"
            )
            
            seed = gr.Number(
                value=0, 
                label="Seed (0 for random)",
                precision=0
            )
            
            status = gr.Textbox(
                label="Status", 
                interactive=False
            )
            
            generate_button = gr.Button("Generate")
            
            # Last Generated Song (audio player)
            audio_output = gr.Audio(
                label="Last Generated Song", 
                interactive=False,
                type="filepath"
            )
            
            # History of Generated Songs
            history = gr.File(
                label="History of Generated Songs (From most Recent to Oldest)",
                file_count="multiple",
                interactive=False
            )
    
    # Event handlers
    generate_button.click(
        fn=run_generation,
        inputs=[genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs],
        outputs=[status, audio_output]
    )

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch YuE Gradio GUI")
    parser.add_argument("--share", action="store_true", default=True, 
                        help="Whether to create a shareable link (default: True)")
    parser.add_argument("--port", type=int, default=None, 
                        help="Port to run the server on (default: Gradio default)")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to bind to (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    # Launch the interface with the specified parameters
    demo.queue().launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port
    )
