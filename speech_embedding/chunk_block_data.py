import json

import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from speech_embedding.paths import project_path


class ChunkBlockSpeechTranscriptDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        speech_model_name,
        summary_tokenizer,
        prompt_text="repeat the speech transcript:",
        sampling_rate=16000,
        chunk_seconds=30,
        max_target_length=128,
        max_prompt_length=32,
        target_field="transcript",
    ):
        self.metadata_path = project_path(metadata_path)
        self.summary_tokenizer = summary_tokenizer
        self.prompt_text = prompt_text
        self.sampling_rate = sampling_rate
        self.chunk_seconds = chunk_seconds
        self.max_target_length = max_target_length
        self.max_prompt_length = max_prompt_length
        self.target_field = target_field
        self.chunk_samples = sampling_rate * chunk_seconds

        self.speech_processor = WhisperProcessor.from_pretrained(speech_model_name)

        self.samples = []
        with self.metadata_path.open("r", encoding="utf-8") as in_file:
            for line in in_file:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def load_audio_chunk(self, item):
        audio_path = project_path(item["audio_path"])
        audio, _ = librosa.load(
            audio_path,
            sr=self.sampling_rate,
            mono=True,
            offset=float(item["start_time"]),
            duration=float(item["duration"]),
        )

        if len(audio) < self.chunk_samples:
            padded = torch.zeros(self.chunk_samples, dtype=torch.float32)
            if len(audio) > 0:
                padded[: len(audio)] = torch.tensor(audio, dtype=torch.float32)
            audio = padded.numpy()

        if len(audio) > self.chunk_samples:
            audio = audio[: self.chunk_samples]

        return audio

    def encode_audio(self, audio):
        speech_inputs = self.speech_processor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        input_features = speech_inputs["input_features"].squeeze(0)
        return input_features.unsqueeze(0)

    def encode_prompt(self):
        prompt = self.summary_tokenizer(
            self.prompt_text,
            max_length=self.max_prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return prompt["input_ids"].squeeze(0), prompt["attention_mask"].squeeze(0)

    def encode_target(self, target_text):
        target = self.summary_tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = target["input_ids"].squeeze(0)
        attention_mask = target["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.summary_tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = self.load_audio_chunk(item)
        input_features = self.encode_audio(audio)
        prompt_input_ids, prompt_attention_mask = self.encode_prompt()
        target_text = item.get(self.target_field, "")
        if not target_text:
            raise ValueError(
                f"Missing target field '{self.target_field}' "
                f"for sample_id={item.get('sample_id', idx)}"
            )
        target_input_ids, target_attention_mask, labels = self.encode_target(target_text)

        return {
            "input_features": input_features,
            "chunk_attention_mask": torch.ones(1, dtype=torch.long),
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels,
        }
