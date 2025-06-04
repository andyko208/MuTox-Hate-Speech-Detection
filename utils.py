import os
import re
import time
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Tuple, Dict
from multiprocessing import Pool, cpu_count

# Specify the paths for the input and output files
MUTOX_ORIG = "mutox.tsv"
MUTOX_INIT = "mutox_init.tsv"
MUTOX_CLEAN = "mutox_clean.tsv"
MUTOX_FAILS = "mutox_fails.tsv"
OUTPUT_DIR = "waveforms/"
CPU_COUNT = cpu_count()*2//3

def get_mutox_init() -> None:
    """
    Removes rows with 1) undefined toxicity labels and 2) missing transcripts
    """
    df = pd.read_csv(MUTOX_ORIG, delimiter='\t')

    # Remove rows with missing transcripts
    no_transcripts = df[pd.isna(df['audio_file_transcript'])]
    df = df.dropna(subset=['audio_file_transcript'])
    df = df[~df.index.isin(no_transcripts.index)]

    # Remove rows with toxicity label undefined
    no_labels = df[df['contains_toxicity'].isin(['Cannot say', 'Cannot Say'])]
    no_labels = df[pd.isna(df['partition'])]
    df = df[~df.index.isin(no_labels.index)]
    df.loc[df['contains_toxicity'] == 'yes', 'partition'] = 'train'
    df.loc[df['contains_toxicity'] == 'yes', 'contains_toxicity'] = 'Yes'
    
    # Export initial version
    df.to_csv(MUTOX_INIT, sep='\t', index=False)


def get_job_params(row: pd.Series) -> Dict:
    """
    Prepare parameters for a single job from a dataframe row.
    
    Args:
        row: Pandas Series containing a single row of data
    
    Returns:
        Dict containing processed parameters
    """
    def sec_to_time(seconds):
        return str(datetime.timedelta(seconds=seconds))
    
    # Parameters for ffmpeg
    url_ss_tt = row['public_url_segment'].split()
    url = url_ss_tt[0]
    start_time = int(url_ss_tt[1]) / 1000
    end_time = int(url_ss_tt[-1]) / 1000
    duration = end_time - start_time
    
    # Parameters for output file
    file_id = row['id']
    lang = row['lang']
    wav_id = f'{lang}_{file_id}'
    wav_url_segment = f'{url} {sec_to_time(start_time)} {sec_to_time(start_time+duration)}'
    wav_output_path = os.path.join(OUTPUT_DIR, f"{wav_id}.wav")
    
    row['id'] = wav_id
    row['public_url_segment'] = f'{wav_url_segment} {wav_output_path}'  # Individually identify locally downloaded waveform
    
    return {
        'row': row,
        'url': url,
        'start_time': start_time,
        'duration': duration,
        'output_path': wav_output_path,
        'exists': os.path.exists(wav_output_path)
    }

def get_waveform(params: Tuple[pd.Series, str, float, float, str]) -> bool:
    """
    Extracts a segment from an audio URL using FFMPEG.
    During the process, remove the corrupted files and log the results
    
    Args:
        params: Tuple containing (row, url, start_time, duration, output_path)
    
    Returns:
        bool: True if successful, False otherwise
    """
    def log_success(row: pd.Series):
        with open(MUTOX_CLEAN, 'a') as f:
            f.write('\t'.join([str(x) for x in row.values]) + '\n')
    def log_fail(id: str, public_url_segment: str, error: str):
        with open(MUTOX_FAILS, 'a') as f:
            # Changed from space to '\t' because mutox_fails.tsv was not recognizing columns
            f.write(f"{id}\t{' '.join(public_url_segment.split()[:-1])}\t{error}\n")
    
    row, url, start_time, duration, output_path = params
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    command = [
        "ffmpeg",
        "-rw_timeout", "10000000",
        "-user_agent", user_agent,
        "-y",
        "-i", url,
        "-ss", str(start_time),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    try:
        result = subprocess.run(command, capture_output=True)    
        if result.returncode != 0:                                                   # ffmpeg failed
            log_fail(row['id'], row['public_url_segment'], f'ffmpeg fail: {result.stderr.decode().splitlines()[-1]}')
        else:                                                                        # ffmpeg succeeded
            if os.path.exists(output_path) and os.path.getsize(output_path) < 10000: # but generated a corrupted file
                os.remove(output_path)
                log_fail(row['id'], row['public_url_segment'], 'corrupted file due to wrong timestamps')
            else:                                                                    # and generated a valid file
                log_success(row)
                return True
    except Exception as e:                                                          # ffmpeg failed
        log_fail(row['id'], row['public_url_segment'], f'ffmpeg exception: {e}')
    return False

def get_mutox_clean() -> None:
    """
    Extract the waveform data from the raw mutox tsv using parallel processing.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_workers = max(1, CPU_COUNT)  
    df = pd.read_csv(MUTOX_INIT, sep="\t")

    jobs = []
    for _, row in df.iterrows():
        
        params = get_job_params(row)
        if not params['exists']:    # Only process if the file does not exist
            jobs.append((params['row'], params['url'], params['start_time'], params['duration'], params['output_path']))
    
    total_jobs = len(jobs)          # Stop if all files are processed
    if total_jobs == 0:
        return
    
    # Process files in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(get_waveform, jobs), total=total_jobs))
    print(f"Total processing: {sum(results)} / {total_jobs}.")


################## Analyzing operations ##################
def plot_durations():
    """Plot the distribution of audio durations in the MuTox dataset."""
    def extract_duration(segment):
        # Regex to extract start and end times
        match = re.search(r'(\d+:\d+:\d+\.\d+|\d+:\d+:\d+)\s+(\d+:\d+:\d+\.\d+|\d+:\d+:\d+)', segment)
        if match:
            start, end = match.groups()
            # Convert to timedelta
            def to_timedelta(t):
                if '.' in t:
                    return pd.to_timedelta(t)
                else:
                    return pd.to_timedelta(t + '.0')
            return (to_timedelta(end) - to_timedelta(start)).total_seconds()
        return None

    df = pd.read_csv('mutox_clean.tsv', delimiter='\t')
    df['duration_sec'] = df['public_url_segment'].apply(extract_duration)
    bins = [0, 1, 2, 3, 4, 5, 10, 20, 60, 120, 300, 600, 10000]
    labels = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-10s', '10-20s', '20-60s', '1-2m', '2-5m', '5-10m', '>10m']
    df['duration_bin'] = pd.cut(df['duration_sec'], bins=bins, labels=labels, right=False)
    grouped = df.groupby(['duration_bin', 'lang']).size().reset_index(name='count')

    pivot = grouped.pivot(index='duration_bin', columns='lang', values='count').fillna(0)
    pivot.plot(kind='bar', stacked=True, figsize=(6,12))
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Duration Range')
    plt.ylabel('Count')
    plt.title('Duration Distribution')
    plt.tight_layout()
    plt.savefig('durations.png')

def get_counts(partition='Total', toxicity=False):
    """Analyze the distribution of valid samples in mutox_clean.tsv"""
    df = pd.read_csv('mutox_clean.tsv', delimiter='\t')

    # Total samples per language
    total_counts = df['lang'].value_counts().sort_index()
    
    # Toxic samples per language
    toxic_counts = df[df['contains_toxicity'] == 'Yes']['lang'].value_counts().sort_index()
    toxic_counts = toxic_counts.reindex(total_counts.index, fill_value=0)

    langs = total_counts.index
    x = np.arange(len(langs))

    plt.figure(figsize=(18, 10))
    plt.bar(x, total_counts, color='skyblue', label='Total samples', width=0.8)
    plt.bar(x, toxic_counts, color='crimson', label='Toxic samples', width=0.8)
    for i, (total, toxic) in enumerate(zip(total_counts, toxic_counts)):
        non_toxic = total - toxic
        y_total = total + max(total_counts) * 0.02
        y_toxic = toxic + max(toxic_counts) * 0.05
        print(f"{langs[i]}: Total: {total}, Toxic: {toxic}")
        plt.text(i, y_total, str(non_toxic), ha='right', va='top', fontsize=10, color='skyblue', fontweight='bold')
        plt.text(i, y_toxic, str(toxic), ha='left', va='bottom', fontsize=10, color='red', fontweight='bold')
        plt.text(i, y_total, str(total), ha='center', va='bottom', fontsize=10, color='black')

    plt.xticks(x, langs, rotation=45, ha='right')
    plt.ylim(0, max(total_counts) * 1.15)
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.title('Sample Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution.png')
    


if __name__ == "__main__":
    """
    Problem with mutox.tsv:
        - Contains rows with missing transcripts
        - Contains rows with undefined toxicity labels
        - Contains rows with timestamps outside the duration of the audio file
    Input: mutox.tsv
    Output: 
        - mutox_clean.tsv: 
            - cleaned version of mutox_initial.tsv
        - mutox_fails.tsv
            - removed rows with wrong timestamps that leads to corrupte waveform file
        - mutox_init.tsv
            - removed rows with undefined toxicity labels
            - removed rows with missing transcripts

        Recommend using tmux or screen to run the script in the background.
    """
    ##################### Obtain the waveform #####################
    # Obtain the initially cleaned version
    get_mutox_init()
    
    # Initialize the clean version and fails file
    pd.read_csv(MUTOX_ORIG, sep='\t', nrows=0).to_csv(MUTOX_CLEAN, sep='\t', index=False)
    with open(MUTOX_FAILS, 'w') as f:
        f.write('\t'.join(['id', 'public_url_segment', 'error']) + '\n')
    f.close()
    
    # Obtain waveforms
    get_mutox_clean()
    
    # Manually remove the ones with "Conversion failed" error from waveforms/
    df = pd.read_csv(MUTOX_CLEAN, sep='\t')
    waveforms = [r[1]['public_url_segment'].split()[-1].split('/')[-1] for r in df.iterrows()]
    for wav in os.listdir(OUTPUT_DIR):
        if wav not in waveforms:
            os.remove(os.path.join(OUTPUT_DIR, wav))

    ##################### Analyze the waveform #####################
    # Plot the sample counts by language and toxicity
    get_counts()
    
    # Plot the duration distribution by language
    plot_durations()
