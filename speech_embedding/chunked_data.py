import json
import math

import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from speech_embedding.paths import project_path


def select_chunk_indices_for_count(num_audio_chunks, max_chunks):
    if num_audio_chunks <= max_chunks:
        return list(range(num_audio_chunks))

    indices = torch.linspace(
        0,
        num_audio_chunks - 1,
        steps=max_chunks,
    )
    return indices.round().long().tolist()


class ChunkedMeetingSpeechSummaryDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        speech_model_name,
        summary_tokenizer,
        prompt_text="summarize the meeting:",
        sampling_rate=16000,
        chunk_seconds=30,
        max_chunks=16,
        max_summary_length=256,
        max_prompt_length=32,
        target_field="summary",
    ):
        self.metadata_path = project_path(metadata_path)
        self.summary_tokenizer = summary_tokenizer
        self.prompt_text = prompt_text
        self.sampling_rate = sampling_rate
        self.chunk_seconds = chunk_seconds
        self.max_chunks = max_chunks
        self.max_summary_length = max_summary_length
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

    def load_audio(self, audio_path):
        audio_path = project_path(audio_path)
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        return audio

    def select_chunk_indices(self, num_audio_chunks):
        return select_chunk_indices_for_count(num_audio_chunks, self.max_chunks)

    def split_audio(self, audio):
        num_audio_chunks = math.ceil(len(audio) / self.chunk_samples)
        selected_indices = self.select_chunk_indices(num_audio_chunks)
        total_chunks = len(selected_indices)

        chunks = []
        for chunk_idx in selected_indices:
            start = chunk_idx * self.chunk_samples
            end = start + self.chunk_samples
            chunk = audio[start:end]

            if len(chunk) < self.chunk_samples:
                padded = torch.zeros(self.chunk_samples, dtype=torch.float32)
                padded[: len(chunk)] = torch.tensor(chunk, dtype=torch.float32)
                chunk = padded.numpy()

            chunks.append(chunk)

        while len(chunks) < self.max_chunks:
            chunks.append(torch.zeros(self.chunk_samples, dtype=torch.float32).numpy())

        chunk_attention_mask = torch.zeros(self.max_chunks, dtype=torch.long)
        chunk_attention_mask[:total_chunks] = 1
        return chunks, chunk_attention_mask

    def encode_chunks(self, chunks):
        features = []
        for chunk in chunks:
            speech_inputs = self.speech_processor(
                chunk,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            features.append(speech_inputs["input_features"].squeeze(0))
        return torch.stack(features, dim=0)

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
            max_length=self.max_summary_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.summary_tokenizer.pad_token_id] = -100
        return labels

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = self.load_audio(item["audio_path"])
        chunks, chunk_attention_mask = self.split_audio(audio)

        input_features = self.encode_chunks(chunks)
        prompt_input_ids, prompt_attention_mask = self.encode_prompt()
        target_text = item.get(self.target_field, "")
        if not target_text:
            raise ValueError(
                f"Missing target field '{self.target_field}' "
                f"for meeting_id={item.get('meeting_id', idx)}"
            )
        labels = self.encode_target(target_text)

        return {
            "input_features": input_features,
            "chunk_attention_mask": chunk_attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "labels": labels,
        }
