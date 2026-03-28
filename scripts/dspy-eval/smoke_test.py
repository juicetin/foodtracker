#!/usr/bin/env python3
"""
Quick smoke test — sends one image through the adb→Gemini Nano bridge
without requiring DSPy. Use this to verify the Android receiver works.

Usage:
    python smoke_test.py <image_path> [device_serial]
"""

import sys
import time
from pathlib import Path

from gemini_nano_lm import (
    GeminiNanoLM,
    _adb_shell,
    _adb_push,
    _adb_rm,
    _adb_file_exists,
    _adb_read_file,
    DEVICE_EVAL_DIR,
    DEVICE_IMAGE_PATH,
    DEVICE_RESULT_PATH,
    DEVICE_DONE_PATH,
    BROADCAST_ACTION,
    RECEIVER_PACKAGE,
    RECEIVER_CLASS,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <image_path> [device_serial]")
        sys.exit(1)

    image_path = sys.argv[1]
    serial = sys.argv[2] if len(sys.argv) > 2 else "48181FDAP00A1U"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Device: {serial}")
    print(f"Image:  {image_path}")

    # Setup
    _adb_shell(serial, f'mkdir -p "{DEVICE_EVAL_DIR}"')

    # Push image
    print("Pushing image to device...")
    _adb_push(serial, image_path, DEVICE_IMAGE_PATH)

    # Clean previous results
    _adb_rm(serial, DEVICE_RESULT_PATH)
    _adb_rm(serial, DEVICE_DONE_PATH)

    # Write prompt to device
    prompt = (
        'Identify all food in this image. Return only valid JSON — no extra text:\n'
        '{"dishes":[{"name":string,"cuisine":string,"recipe_name":string,'
        '"ingredients":[{"name":string,"amount_g":number}]}]}\n'
        'recipe_name: a concise human-friendly name for the dish as a recipe '
        '(e.g. "Chicken Stir Fry with Vegetables"). '
        'Estimate amount_g using surrounding objects (plates, cutlery, cups, hands) '
        'as size references; fall back to a typical restaurant serving size if no '
        'reference objects are visible. '
        'Be specific with ingredient names (e.g. "basmati rice" not "rice").'
    )

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_local = f.name
    _adb_push(serial, prompt_local, f"{DEVICE_EVAL_DIR}/prompt.txt")
    os.unlink(prompt_local)

    # Fire broadcast
    print("Sending broadcast to trigger Gemini Nano...")
    start = time.time()

    cmd = (
        f'nohup am broadcast -a {BROADCAST_ACTION} '
        f'--es image_path "{DEVICE_IMAGE_PATH}" '
        f'--es prompt_path "{DEVICE_EVAL_DIR}/prompt.txt" '
        f'--es result_path "{DEVICE_RESULT_PATH}" '
        f'{RECEIVER_PACKAGE}/{RECEIVER_CLASS} > /dev/null 2>&1 &'
    )
    broadcast_result = _adb_shell(serial, cmd, timeout=5)
    print(f"Broadcast result: {broadcast_result}")

    # Poll for completion
    print("Waiting for result...", end="", flush=True)
    elapsed = 0.0
    while elapsed < 30.0:
        if _adb_file_exists(serial, DEVICE_DONE_PATH):
            break
        print(".", end="", flush=True)
        time.sleep(0.5)
        elapsed += 0.5
    print()

    duration = time.time() - start

    if not _adb_file_exists(serial, DEVICE_DONE_PATH):
        print(f"TIMEOUT after {duration:.1f}s — no result from device")
        print("\nTroubleshooting:")
        print("  1. Is the app installed and running?")
        print("  2. Is Gemini Nano available? Check: adb shell 'pm list packages | grep aicore'")
        print("  3. Check logcat: adb logcat -s VlmEvalReceiver")
        sys.exit(1)

    # Read result
    result = _adb_read_file(serial, DEVICE_RESULT_PATH)
    print(f"\n{'='*60}")
    print(f"Result ({duration:.1f}s):")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}")

    if result.startswith("ERROR:"):
        print(f"\nGemini Nano returned an error. Check logcat for details:")
        print(f"  adb -s {serial} logcat -s GeminiNano VlmEvalReceiver")
    else:
        # Try to parse and pretty-print
        try:
            import json
            parsed = json.loads(result)
            print(f"\nParsed JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(f"\nNote: Response is not valid JSON (may be truncated)")


if __name__ == "__main__":
    main()
