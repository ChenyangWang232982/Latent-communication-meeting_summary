import json

import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from speech_embedding.paths import project_path

class MeetingSpeechSummaryDataset(Dataset):
    def __init__(self, metadata_path, speech_model_name, summary_tokenizer, prompt_text="summarize the meeting:", sampling_rate=16000, max_summary_length=128,max_prompt_length=32):
        self.metadata_path = project_path(metadata_path)
        self.summary_tokenizer = summary_tokenizer
        self.prompt_text = prompt_text
        self.sampling_rate = sampling_rate
        self.max_summary_length = max_summary_length
        self.max_prompt_length = max_prompt_length

        self.speech_processor = WhisperProcessor.from_pretrained(
            speech_model_name
        )

        self.samples = []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def load_audio(self, audio_path):
        audio_path = project_path(audio_path)
        audio, _ = librosa.load(
            audio_path,
            sr = self.sampling_rate,
            mono = True
        )
        return audio

    def __getitem__(self, idx):
        item = self.samples[idx]

        audio = self.load_audio(item["audio_path"])

        speech_inputs = self.speech_processor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        input_features = speech_inputs["input_features"].squeeze(0)

        prompt = self.summary_tokenizer(
            self.prompt_text,
            max_length = self.max_prompt_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )

        prompt_input_ids = prompt["input_ids"].squeeze(0)
        prompt_attention_mask = prompt["attention_mask"].squeeze(0)

        target = self.summary_tokenizer(
            item["summary"],
            max_length = self.max_summary_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.summary_tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "labels": labels,
        }




