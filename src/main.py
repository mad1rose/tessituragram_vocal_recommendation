"""CLI entry point for tessituragram analysis."""

import argparse
import sys
from pathlib import Path

from src.parser import extract_vocal_line
from src.tessituragram import generate_tessituragram, calculate_statistics
from src.metadata import extract_metadata
from src.storage import save_tessituragrams, load_tessituragrams, merge_songs


def process_file(filepath: Path) -> dict:
    """
    Process a single MusicXML file and generate tessituragram.
    
    Args:
        filepath: Path to .mxl file
        
    Returns:
        Dictionary with composer, title, filename, tessituragram, and statistics
    """
    print(f"Processing: {filepath.name}")
    
    try:
        # Extract metadata
        metadata = extract_metadata(filepath)
        
        # Extract vocal line
        notes = extract_vocal_line(filepath)
        
        if not notes:
            print(f"  Warning: No notes found in {filepath.name}")
            return None
        
        # Generate tessituragram
        tessituragram = generate_tessituragram(notes)
        
        # Calculate statistics
        statistics = calculate_statistics(notes, tessituragram)
        
        return {
            'composer': metadata['composer'],
            'title': metadata['title'],
            'filename': filepath.name,
            'tessituragram': tessituragram,
            'statistics': statistics
        }
    
    except Exception as e:
        print(f"  Error processing {filepath.name}: {e}")
        return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate tessituragrams from MusicXML files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='mxl_songs',
        help='Directory containing .mxl files (default: mxl_songs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/tessituragrams.json',
        help='Output JSON file path (default: data/tessituragrams.json)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process a single file instead of all files in directory'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    songs = []
    
    if args.file:
        # Process single file
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        
        result = process_file(filepath)
        if result:
            songs.append(result)
    else:
        # Process all .mxl files in directory
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)
        
        mxl_files = list(input_dir.glob('*.mxl'))
        
        if not mxl_files:
            print(f"Error: No .mxl files found in {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(mxl_files)} .mxl file(s)")
        print()
        
        for filepath in sorted(mxl_files):
            result = process_file(filepath)
            if result:
                songs.append(result)
    
    if not songs:
        print("Error: No songs processed successfully")
        sys.exit(1)
    
    # Load existing library (if any) and merge to avoid duplicates
    existing: list[dict] = []
    if output_path.exists():
        existing = load_tessituragrams(output_path)

    merged = merge_songs(existing, songs)
    new_count = len(merged) - len(existing)

    print(f"\n{new_count} new song(s) added ({len(existing)} already in library)")
    print(f"Saving {len(merged)} total tessituragram(s) to {output_path}")
    save_tessituragrams(merged, output_path)
    print("Done!")


if __name__ == '__main__':
    main()
