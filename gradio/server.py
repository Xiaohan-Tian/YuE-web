import gradio as gr
import threading
import time

# Global variable to control the execution
abort_flag = False

def mock_generate(genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs):
    global abort_flag
    abort_flag = False
    print("mock_generate has been invoked!")
    
    # Update status with progress
    status_updates = []
    # Simulate a long-running process
    for i in range(10):
        if abort_flag:
            print("Generation aborted!")
            return "Aborted", None
        # Simulate work
        time.sleep(0.5)  # Simulate work time
        print(f"Generating... {i+1}/10")
        status_updates.append(f"Generating... {i+1}/10")
        yield ", ".join(status_updates), None
    
    # In a real implementation, this would generate an audio file
    dummy_audio = "https://audio-samples.github.io/samples/mp3/blizzard_biased/sample-1.mp3"
    
    return "Generation complete!", dummy_audio

def abort():
    global abort_flag
    abort_flag = True
    return "Aborting...", None

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
                lines=3
            )
            
            lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="[verse]\nStaring at the sunset, colors paint the sky\nThoughts of you keep swirling, can't deny\nI know I let you down, I made mistakes\nBut I'm here to mend the heart I didn't break\n\n[chorus]\nEvery road you take, I'll be one step behind\nEvery dream you chase, I'm reaching for the light",
                lines=30
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
            abort_button = gr.Button("Abort")
            
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
        fn=mock_generate,
        inputs=[genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs],
        outputs=[status, audio_output]
    )
    
    abort_button.click(
        fn=abort,
        inputs=None,
        outputs=[status, audio_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.queue().launch()
