import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
from datasets import load_dataset

youtube_dataset = load_dataset("lion-ai/youtube_asr_30", split="train")
gemini_dataset = load_dataset("lion-ai/pl_med_asr_test2", split="train")
admed_anoni_dataset = load_dataset("lion-ai/admed_voice", split="anoni")
admed_human_dataset = load_dataset("lion-ai/admed_voice", split="human")

prepared_data_dir = os.environ.get("PREPARED_DATA_DIR", "prepared_data")
processed_data_dir = os.environ.get("PROCESSED_DATA_DIR", "processed_data")
