import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../inference/xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../inference/xcodec_mini_infer', 'descriptaudiocodec'))
import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf
import tempfile

# Import modules from inference directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../inference'))
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched

def generate(genre_prompt, lyrics, num_sequences, num_tokens, seed, num_songs):
    # Log input values
    print("Genre Prompt:", genre_prompt)
    print("Lyrics:", lyrics)
    print("Number of Sequences:", num_sequences)
    print("Number of tokens per sequence:", num_tokens)
    print("Seed:", seed)
    print("Number of Songs:", num_songs)
    print("Inference has started!")
    
    # Set fixed parameters
    cuda_idx = 0
    stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"
    stage2_model = "m-a-p/YuE-s2-1B-general"
    stage2_batch_size = 4
    output_dir = "../output"
    max_new_tokens = num_tokens
    repetition_penalty = 1.1
    run_n_segments = num_sequences
    
    # Create temp files for genre and lyrics
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as genre_file:
        genre_file.write(genre_prompt)
        genre_txt = genre_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as lyrics_file:
        lyrics_file.write(lyrics)
        lyrics_txt = lyrics_file.name
    
    # Setup output directories
    stage1_output_dir = os.path.join(output_dir, "stage1")
    stage2_output_dir = os.path.join(output_dir, "stage2")
    os.makedirs(stage1_output_dir, exist_ok=True)
    os.makedirs(stage2_output_dir, exist_ok=True)
    
    # Seed everything
    def seed_everything(seed=42): 
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    seed_everything(seed if seed != 0 else random.randint(1, 10000))
    
    # Setup device
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    mmtokenizer = _MMSentencePieceTokenizer("../inference/mm_tokenizer_v0.2_hf/tokenizer.model")
    model = AutoModelForCausalLM.from_pretrained(
        stage1_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)
    
    # Setup codec tools
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)
    model_config = OmegaConf.load('../inference/xcodec_mini_infer/final_ckpt/config.yaml')
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
    parameter_dict = torch.load('../inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', map_location='cpu', weights_only=False)
    codec_model.load_state_dict(parameter_dict['codec_model'])
    codec_model.to(device)
    codec_model.eval()
    
    # Define helper classes and functions
    class BlockTokenRangeProcessor(LogitsProcessor):
        def __init__(self, start_id, end_id):
            self.blocked_token_ids = list(range(start_id, end_id))

        def __call__(self, input_ids, scores):
            scores[:, self.blocked_token_ids] = -float("inf")
            return scores
    
    def load_audio_mono(filepath, sampling_rate=16000):
        audio, sr = torchaudio.load(filepath)
        # Convert to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample if needed
        if sr != sampling_rate:
            resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
        if len(audio_prompt.shape) < 3:
            audio_prompt.unsqueeze_(0)
        with torch.no_grad():
            raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int16)
        return raw_codes

    def split_lyrics(lyrics):
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

    # Stage 1 inference
    stage1_output_set = []
    
    # Load genre and lyrics
    with open(genre_txt) as f:
        genres = f.read().strip()
    with open(lyrics_txt) as f:
        lyrics_content = f.read()
    lyrics = split_lyrics(lyrics_content)
    
    # Prepare prompt
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics
    
    random_id = uuid.uuid4()
    output_seq = None
    
    # Decoding config
    top_p = 0.93
    temperature = 1.0
    
    # Special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    
    # Format text prompt
    run_n_segments = min(run_n_segments+1, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <=1 else 1.2
        if i==0:
            continue
        if i==1:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384-max_new_tokens-1
        if input_ids.shape[-1] > max_context:
            print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
            
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=100, 
                do_sample=True, 
                top_p=top_p,
                temperature=temperature, 
                repetition_penalty=repetition_penalty, 
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
            )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
                
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # Save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx)!=len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    range_begin = 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
        
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    
    vocal_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
    inst_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
    
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)

    # Offload model
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # Stage 2 inference
    print("Stage 2 inference...")
    model_stage2 = AutoModelForCausalLM.from_pretrained(
        stage2_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model_stage2.to(device)
    model_stage2.eval()

    if torch.__version__ >= "2.0.0":
        model_stage2 = torch.compile(model_stage2)

    def stage2_generate(model, prompt, batch_size=16):
        codec_ids = codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
                        codec_ids, 
                        global_offset=codectool.global_offset, 
                        codebook_size=codectool.codebook_size, 
                        num_codebooks=codectool.num_codebooks, 
                    ).astype(np.int32)
        
        # Prepare prompt_ids based on batch size or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(device)
        prompt_ids = torch.as_tensor(prompt_ids).to(device)
        len_prompt = prompt_ids.shape[-1]
        
        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

        # Teacher forcing generate loop
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx:frames_idx+1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = model.generate(input_ids=input_ids, 
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )
            
            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
        stage2_result = []
        for i in tqdm(range(len(stage1_output_set))):
            output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
            
            if os.path.exists(output_filename):
                print(f'{output_filename} stage2 has done.')
                continue
            
            # Load the prompt
            prompt = np.load(stage1_output_set[i]).astype(np.int32)
            
            # Only accept 6s segments
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6
            
            if num_batch <= batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                segments = []
                num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

                for seg in range(num_segments):
                    start_idx = seg * batch_size * 300
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                    current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                    segment = stage2_generate(
                        model,
                        prompt[:, start_idx:end_idx],
                        batch_size=current_batch_size
                    )
                    segments.append(segment)

                # Concatenate all the segments
                output = np.concatenate(segments, axis=0)
            
            # Process the ending part of the prompt
            if output_duration*50 != prompt.shape[-1]:
                ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
                output = np.concatenate([output, ending], axis=0)
            output = codectool_stage2.ids2npy(output)

            # Fix invalid codes (a dirty solution, which may harm the quality of audio)
            # We are trying to find better one
            fixed_output = copy.deepcopy(output)
            for i, line in enumerate(output):
                for j, element in enumerate(line):
                    if element < 0 or element > 1023:
                        counter = Counter(line)
                        most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                        fixed_output[i, j] = most_frequant
            # save output
            np.save(output_filename, fixed_output)
            stage2_result.append(output_filename)
        return stage2_result

    # Run stage 2 inference
    stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=stage2_batch_size)
    print(stage2_result)
    print('Stage 2 DONE.\n')
    
    # Audio conversion and processing
    def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
        torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
    
    # Reconstruct tracks
    recons_output_dir = os.path.join(output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, 'mix')
    os.makedirs(recons_mix_dir, exist_ok=True)
    tracks = []
    for npy in stage2_result:
        codec_result = np.load(npy)
        decodec_rlt=[]
        with torch.no_grad():
            decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)
    
    # Mix tracks
    for inst_path in tracks:
        try:
            if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                and '_itrack' in inst_path:
                # find pair
                vocal_path = inst_path.replace('_itrack', '_vtrack')
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)
    
    # Vocoder to upsample audios
    vocal_decoder, inst_decoder = build_codec_model('../inference/xcodec_mini_infer/decoders/config.yaml', 
                                                   '../inference/xcodec_mini_infer/decoders/decoder_131000.pth', 
                                                   '../inference/xcodec_mini_infer/decoders/decoder_151000.pth')
    vocoder_output_dir = os.path.join(output_dir, 'vocoder')
    vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
    vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    
    for npy in stage2_result:
        if '_itrack' in npy:
            # Process instrumental
            instrumental_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, 'itrack.mp3'),
                False,  # rescale
                None,   # args
                inst_decoder,
                codec_model
            )
        else:
            # Process vocal
            vocal_output = process_audio(
                npy,
                os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
                False,  # rescale
                None,   # args 
                vocal_decoder,
                codec_model
            )
    
    # Mix tracks
    try:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, False)  # rescale=False
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(f"mix failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")
    
    # Post process
    final_output = os.path.join(output_dir, os.path.basename(recons_mix))
    replace_low_freq_with_energy_matched(
        a_file=recons_mix,     # 16kHz
        b_file=vocoder_mix,    # 48kHz
        c_file=final_output,
        cutoff_freq=5500.0
    )
    
    # Clean up temp files
    os.unlink(genre_txt)
    os.unlink(lyrics_txt)
    
    print("Inference is done!")
    print(f"Output file: {final_output}")
    
    return final_output
