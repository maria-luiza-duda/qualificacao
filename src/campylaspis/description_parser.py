"""
Structured Description Parser for Campylaspis Body-Part Aware Generation

This module parses species description text files and extracts structured descriptions
for specific body parts, enabling anatomically accurate prompt generation.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict


class DescriptionParser:
    """
    Parser for extracting structured body part descriptions from species text files.
    """

    # Body parts to extract
    BODY_PARTS = [
        'carapace',
        'pseudorostrum',
        'pereopod_1',
        'pereopod_2',
        'pereopod_3',
        'pereopod_4',
        'pereopod_5',
        'uropod',
        'whole_body'
    ]

    # Keywords for fallback mapping
    FALLBACK_MAPPING = {
        'pereopod_1': ['pereopod', 'first pereopod', 'pereopods'],
        'pereopod_2': ['pereopod', 'second pereopod', 'pereopods'],
        'pereopod_3': ['pereopod', 'third pereopod', 'pereopods'],
        'pereopod_4': ['pereopod', 'fourth pereopod', 'pereopods'],
        'pereopod_5': ['pereopod', 'fifth pereopod', 'pereopods'],
        'carapace': ['carapace', 'cephalothorax'],
        'pseudorostrum': ['rostrum', 'pseudorostrum'],
        'uropod': ['uropod', 'uropods', 'telson'],
        'whole_body': ['body', 'morphology', 'description']
    }

    def __init__(self, text_root: str = "datasets/campylaspis"):
        """
        Initialize the parser with the root directory for text files.

        Args:
            text_root: Root directory containing species folders with text files
        """
        self.text_root = Path(text_root)

    def parse_species_description(self, species: str) -> Dict[str, str]:
        """
        Parse description file for a species and extract structured body part descriptions.

        Args:
            species: Species name (e.g., 'aculeata')

        Returns:
            Dictionary mapping body parts to their descriptions
        """
        text_file = self.text_root / species / "text" / f"{species}.txt"

        if not text_file.exists():
            print(f"⚠️  Description file not found: {text_file}")
            return self._get_empty_descriptions()

        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()

            return self._extract_body_part_descriptions(content)

        except Exception as e:
            print(f"❌ Error parsing description for {species}: {e}")
            return self._get_empty_descriptions()

    def _extract_body_part_descriptions(self, content: str) -> Dict[str, str]:
        """
        Extract descriptions for each body part from the text content.

        Args:
            content: Full text content of the species description

        Returns:
            Dictionary of body part descriptions
        """
        descriptions = {}

        # Convert to lowercase for case-insensitive matching
        content_lower = content.lower()

        for part in self.BODY_PARTS:
            description = self._extract_part_description(content, content_lower, part)
            descriptions[part] = description

        return descriptions

    def _extract_part_description(self, content: str, content_lower: str, body_part: str) -> str:
        """
        Extract description for a specific body part using various strategies.

        Args:
            content: Original text content
            content_lower: Lowercase version for matching
            body_part: Body part to extract (e.g., 'pereopod_1')

        Returns:
            Extracted description or fallback
        """
        # Strategy 1: Direct section headers
        section_patterns = [
            rf'(?:{body_part}){{.*?}}(.*?)(?=\n\n|\n[A-Z]|\Z)',
            rf'(?:{body_part.replace("_", " ")}){{.*?}}(.*?)(?=\n\n|\n[A-Z]|\Z)',
        ]

        for pattern in section_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                description = match.group(1).strip()
                if len(description) > 20:  # Minimum length check
                    return self._clean_description(description)

        # Strategy 2: Keyword-based extraction
        keywords = self.FALLBACK_MAPPING.get(body_part, [body_part.replace('_', ' ')])
        sentences = re.split(r'[.!?]+', content)

        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            combined = ' '.join(relevant_sentences[:3])  # Take first 3 relevant sentences
            return self._clean_description(combined)

        # Strategy 3: Fallback to general description
        return self._get_fallback_description(body_part)

    def _clean_description(self, description: str) -> str:
        """
        Clean and normalize extracted description text.

        Args:
            description: Raw description text

        Returns:
            Cleaned description
        """
        # Remove extra whitespace
        description = re.sub(r'\s+', ' ', description)

        # Remove citation markers and references
        description = re.sub(r'\[\d+\]', '', description)
        description = re.sub(r'\(\w+\s+\d{4}\)', '', description)

        # Capitalize first letter
        description = description.strip()
        if description:
            description = description[0].upper() + description[1:]

        return description

    def _get_fallback_description(self, body_part: str) -> str:
        """
        Provide fallback descriptions when specific text is not found.

        Args:
            body_part: Body part identifier

        Returns:
            Generic fallback description
        """
        fallbacks = {
            'carapace': 'Dorsal shield covering the cephalothorax, typically smooth or with slight ornamentation.',
            'pseudorostrum': 'Anterior projection from the carapace, varying in shape and length among species.',
            'pereopod_1': 'First pereopod, typically modified for grasping or sensory functions.',
            'pereopod_2': 'Second pereopod, used for locomotion and substrate interaction.',
            'pereopod_3': 'Third pereopod, contributing to ambulatory movement.',
            'pereopod_4': 'Fourth pereopod, involved in walking and substrate contact.',
            'pereopod_5': 'Fifth pereopod, often reduced or modified for specific functions.',
            'uropod': 'Posterior appendages, typically biramous and involved in swimming or burrowing.',
            'whole_body': 'Small marine crustacean with elongated body, typically 2-5mm in length.'
        }

        return fallbacks.get(body_part, f'Anatomical structure of the {body_part.replace("_", " ")}.')

    def _get_empty_descriptions(self) -> Dict[str, str]:
        """
        Return empty descriptions dictionary for error cases.

        Returns:
            Dictionary with empty strings for all body parts
        """
        return {part: "" for part in self.BODY_PARTS}

    def get_description_summary(self, species: str) -> Dict[str, any]:
        """
        Get a summary of available descriptions for a species.

        Args:
            species: Species name

        Returns:
            Summary dictionary with description lengths and completeness
        """
        descriptions = self.parse_species_description(species)

        summary = {
            'species': species,
            'total_parts': len(descriptions),
            'parts_with_descriptions': sum(1 for desc in descriptions.values() if desc),
            'description_lengths': {part: len(desc) for part, desc in descriptions.items()},
            'completeness_percentage': round(sum(1 for desc in descriptions.values() if desc) / len(descriptions) * 100, 1)
        }

        return summary


def parse_species_descriptions(species_list: List[str], text_root: str = "datasets/campylaspis") -> Dict[str, Dict[str, str]]:
    """
    Parse descriptions for multiple species.

    Args:
        species_list: List of species names
        text_root: Root directory for text files

    Returns:
        Dictionary mapping species to their body part descriptions
    """
    parser = DescriptionParser(text_root)
    all_descriptions = {}

    for species in species_list:
        descriptions = parser.parse_species_description(species)
        all_descriptions[species] = descriptions

    return all_descriptions


def print_description_summary(species: str, text_root: str = "datasets/campylaspis"):
    """
    Print a summary of descriptions for a species.

    Args:
        species: Species name
        text_root: Root directory for text files
    """
    parser = DescriptionParser(text_root)
    summary = parser.get_description_summary(species)

    print(f"📝 Description Summary for {species}")
    print("=" * 40)
    print(f"Completeness: {summary['completeness_percentage']}%")
    print(f"Parts with descriptions: {summary['parts_with_descriptions']}/{summary['total_parts']}")

    descriptions = parser.parse_species_description(species)
    print("\n📋 Body part descriptions:")
    for part, desc in descriptions.items():
        status = "✅" if desc else "❌"
        length = len(desc)
        preview = desc[:60] + "..." if len(desc) > 60 else desc
        print(f"{status} {part}: ({length} chars) {preview}")


if __name__ == "__main__":
    # Example usage
    species = "aculeata"

    try:
        print_description_summary(species)

        # Parse specific description
        parser = DescriptionParser()
        descriptions = parser.parse_species_description(species)
        print(f"\n🔍 Pereopod_1 description: {descriptions.get('pereopod_1', 'Not found')}")

    except Exception as e:
        print(f"❌ Error: {e}")