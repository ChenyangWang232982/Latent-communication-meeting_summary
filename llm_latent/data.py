import json
from pathlib import Path

from torch.utils.data import Dataset 

class MeetingLatentDataset(Dataset):
    def __init__(
            self,
            metadata_path,
            tokenizer,
            sender_max_length=256,
            receiver_max_length=32,
            target_max_length=64
    ):
        self.tokenizer = tokenizer
        self.sender_max_length = sender_max_length
        self.receiver_max_length = receiver_max_length
        self.target_max_length = target_max_length

        self.samples = []
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        # Chunk-level files have teacher_transcript; meeting-level AMI splits
        # normally store the source text under transcript.
        transcript = item.get("teacher_transcript") or item.get("transcript")
        if not transcript:
            raise ValueError(
                f"Missing transcript text for sample {item.get('meeting_id', index)}"
            )
        target = item["summary"]

        sender_text = (
            "Read the meeting transcript and encode its key information: "
            + transcript
        )

        receiver_prompt = (
            "Write a concise meeting summary based on the received information:"
        )

        sender_inputs = self.tokenizer(
            sender_text,
            max_length=self.sender_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        receiver_inputs = self.tokenizer(
            receiver_prompt,
            max_length=self.receiver_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_inputs = self.tokenizer(
            target,
            max_length=self.target_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "sender_input_ids": sender_inputs["input_ids"].squeeze(0),
            "sender_attention_mask": sender_inputs["attention_mask"].squeeze(0),
            "receiver_input_ids": receiver_inputs["input_ids"].squeeze(0),
            "receiver_attention_mask": receiver_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }
