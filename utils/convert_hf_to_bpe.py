#!/usr/bin/env python3
import json
import struct
import argparse
from pathlib import Path

MAGIC = b"BPE1"
FORMAT_VERSION = 1


def build_unicode_to_bytes_map() -> dict[str, int]:
    """
    Reconstructs OpenAI's byte_to_unicode mapping in reverse.
    GPT-2 maps 256 byte values to printable unicode characters so that
    whitespace/control bytes do not break text-based BPE files.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # Map from the unicode character string back to raw uint8 byte value
    return {chr(c): b for b, c in zip(bs, cs)}


def token_str_to_raw_bytes(token_str: str, unichar_to_byte: dict[str, int]) -> bytes:
    """
    Decodes an escaped GPT-2 unicode string back to its exact raw byte representation.
    """
    raw = bytearray()
    for ch in token_str:
        if ch in unichar_to_byte:
            raw.append(unichar_to_byte[ch])
        else:
            # Fallback for special tokens like <|endoftext|> which are standard UTF-8
            raw.extend(ch.encode("utf-8"))
    return bytes(raw)


def convert(vocab_path: Path, merges_path: Path, output_path: Path):
    print(f"[*] Loading vocab from: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json: dict[str, int] = json.load(f)

    unichar_to_byte = build_unicode_to_bytes_map()

    # Map: TokenId -> raw bytes
    # Map: Raw Bytes -> TokenId
    id_to_bytes: dict[int, bytes] = {}
    bytes_to_id: dict[bytes, int] = {}

    for token_str, token_id in vocab_json.items():
        raw_b = token_str_to_raw_bytes(token_str, unichar_to_byte)
        id_to_bytes[token_id] = raw_b
        bytes_to_id[raw_b] = token_id

    # Ensure all 256 base bytes exist in the vocabulary
    for b in range(256):
        single_byte = bytes([b])
        if single_byte not in bytes_to_id:
            # Assign if somehow omitted (standard GPT-2 vocab contains all 256)
            new_id = len(id_to_bytes)
            id_to_bytes[new_id] = single_byte
            bytes_to_id[single_byte] = new_id

    print(f"[*] Total vocabulary size: {len(id_to_bytes)} tokens")

    # Parse merges.txt
    print(f"[*] Parsing merge rules from: {merges_path}")
    merges: list[tuple[int, int, int, int]] = []  # (left_id, right_id, rank, result_id)

    with open(merges_path, "r", encoding="utf-8") as f:
        rank = 0
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip comments/version header

            parts = line.split()
            if len(parts) != 2:
                continue

            left_str, right_str = parts[0], parts[1]
            left_bytes = token_str_to_raw_bytes(left_str, unichar_to_byte)
            right_bytes = token_str_to_raw_bytes(right_str, unichar_to_byte)
            merged_bytes = left_bytes + right_bytes

            # Check validity in vocabulary
            if left_bytes not in bytes_to_id or right_bytes not in bytes_to_id:
                continue

            left_id = bytes_to_id[left_bytes]
            right_id = bytes_to_id[right_bytes]

            if merged_bytes in bytes_to_id:
                result_id = bytes_to_id[merged_bytes]
                merges.append((left_id, right_id, rank, result_id))
                rank += 1

    print(f"[*] Total valid merge rules: {len(merges)}")

    # Write custom C++ binary format
    print(f"[*] Serializing to binary file: {output_path}")
    with open(output_path, "wb") as out:
        # 1. Header: Magic (4B) + Version (1B) + Vocab Size (uint32) + Merge Count (uint32)
        out.write(MAGIC)
        out.write(struct.pack("<B", FORMAT_VERSION))
        out.write(struct.pack("<I", len(id_to_bytes)))
        out.write(struct.pack("<I", len(merges)))

        # 2. Vocabulary Table: Repeated for each token
        #    Token ID (uint32) + Byte Length (uint32) + Raw Bytes
        for token_id, raw_b in id_to_bytes.items():
            out.write(struct.pack("<I", token_id))
            out.write(struct.pack("<I", len(raw_b)))
            out.write(raw_b)

        # 3. Merges Table: Repeated for each merge rule
        #    Left ID (uint32) + Right ID (uint32) + Rank (uint32) + Result ID (uint32)
        for left_id, right_id, m_rank, result_id in merges:
            out.write(struct.pack("<IIII", left_id, right_id, m_rank, result_id))

    file_size_kb = output_path.stat().st_size / 1024
    print(f"[+] Done! Successfully created '{output_path}' ({file_size_kb:.2f} KB).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face GPT-2 vocab.json and merges.txt to C++ BPE binary format."
    )
    parser.add_argument("--vocab", type=Path, default="gpt2_tokenizer/vocab.json", help="Path to vocab.json")
    parser.add_argument("--merges", type=Path, default="gpt2_tokenizer/merges.txt", help="Path to merges.txt")
    parser.add_argument("--out", type=Path, default="gpt2.bpe", help="Output .bpe binary file path")

    args = parser.parse_args()
    convert(args.vocab, args.merges, args.out)