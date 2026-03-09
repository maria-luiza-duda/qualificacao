"""
Morphology Parser for Scientific Image Generation

This module parses taxonomic descriptions into structured morphological constraints
for generating scientifically accurate organism images.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import re


class MorphologyParser:
    """
    Parses taxonomic descriptions into structured morphological constraints.
    
    Converts free-form taxonomic text into structured constraints that can be
    used to guide image generation with strict morphological fidelity.
    """
    
    def __init__(self):
        """Initialize the morphology parser."""
        self._clip_tokenizer = None

        self.body_part_aliases = {
            'carapace': 'carapace',
            'pseudorostrum': 'pseudorostrum',
            'eyelobe': 'eyelobe',
            'eye lobe': 'eyelobe',
            'eyes': 'eyelobe',
            'eye': 'eyelobe',
            'antennule': 'antenna_1',
            'antennules': 'antenna_1',
            'antenna 1': 'antenna_1',
            'antenna i': 'antenna_1',
            'uropod': 'uropod',
            'uropods': 'uropod',
            'pleon': 'pleon_abdomen',
            'abdomen': 'pleon_abdomen',
            'pereon': 'pereon_thorax',
            'thorax': 'pereon_thorax',
            'maxilliped 2': 'maxilliped_2',
            'maxilliped ii': 'maxilliped_2',
            'maxilliped 3': 'maxilliped_3',
            'maxilliped iii': 'maxilliped_3',
            'pereopod 1': 'pereopod_1',
            'pereopod i': 'pereopod_1',
            'pereopod 2': 'pereopod_2',
            'pereopod ii': 'pereopod_2',
            'pereopod 3': 'pereopod_3',
            'pereopod iii': 'pereopod_3',
            'pereopod 4': 'pereopod_4',
            'pereopod iv': 'pereopod_4',
            'pereopod 5': 'pereopod_5',
            'pereopod v': 'pereopod_5',
        }

        self.constraint_display_labels = {
            'carapace': 'Carapace',
            'pseudorostrum': 'Pseudorostrum',
            'eyelobe': 'Eyelobe',
            'antenna_1': 'Antenna 1 / Antennule',
            'maxilliped_2': 'Maxilliped 2',
            'maxilliped_3': 'Maxilliped 3',
            'pereopod_1': 'Pereopod 1',
            'pereopod_2': 'Pereopod 2',
            'pereopod_3': 'Pereopod 3',
            'pereopod_4': 'Pereopod 4',
            'pereopod_5': 'Pereopod 5',
            'uropod': 'Uropod',
            'pleon_abdomen': 'Pleon / Abdomen',
            'pereon_thorax': 'Pereon / Thorax',
        }

        self.constraint_order = [
            'carapace',
            'pseudorostrum',
            'eyelobe',
            'antenna_1',
            'maxilliped_2',
            'maxilliped_3',
            'pereopod_1',
            'pereopod_2',
            'pereopod_3',
            'pereopod_4',
            'pereopod_5',
            'uropod',
            'pleon_abdomen',
            'pereon_thorax',
        ]

        self.priority_keys = [
            'carapace',
            'pseudorostrum',
            'pereopod_2',
            'uropod',
            'maxilliped_3',
            'maxilliped_2',
            'eyelobe',
            'antenna_1',
            'pereopod_1',
        ]
        
        # Essential priority parts that should ALWAYS be included if present
        self.essential_priority_keys = [
            'carapace',
            'pseudorostrum',
            'pereopod_2',
            'uropod',
        ]
    
    def parse_taxonomic_description(self, description_text: str) -> Dict[str, str]:
        """
        Parse taxonomic description text into structured morphological constraints.

        Args:
            description_text: Raw taxonomic description

        Returns:
            Dictionary mapping body parts to original extracted phrases
        """
        if not description_text or not description_text.strip():
            return {}

        constraints: Dict[str, str] = {}
        sections = self._extract_sections(description_text)

        for raw_label, raw_phrase in sections:
            canonical_key = self._canonicalize_body_part(raw_label)
            if canonical_key is None:
                continue

            cleaned_phrase = self._clean_phrase(raw_phrase)
            if not cleaned_phrase:
                continue

            if canonical_key in constraints:
                if cleaned_phrase not in constraints[canonical_key]:
                    constraints[canonical_key] = f"{constraints[canonical_key]} {cleaned_phrase}".strip()
            else:
                constraints[canonical_key] = cleaned_phrase

        return constraints

    def _extract_sections(self, description_text: str) -> List[Tuple[str, str]]:
        """Extract labeled sections like 'Carapace: ...' while preserving full phrases."""
        text = description_text.replace('\r\n', '\n').replace('\r', '\n').strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        sections: List[Tuple[str, str]] = []
        current_label: Optional[str] = None
        current_chunks: List[str] = []

        for line in lines:
            if ':' in line:
                candidate_label, candidate_phrase = line.split(':', 1)
                if self._looks_like_label(candidate_label):
                    if current_label is not None:
                        sections.append((current_label, ' '.join(chunk for chunk in current_chunks if chunk).strip()))
                    current_label = candidate_label.strip()
                    current_chunks = [candidate_phrase.strip()]
                    continue

            if current_label is not None:
                current_chunks.append(line)

        if current_label is not None:
            sections.append((current_label, ' '.join(chunk for chunk in current_chunks if chunk).strip()))

        if sections:
            return sections

        fallback_sections: List[Tuple[str, str]] = []
        marker_pattern = re.compile(r'([A-Za-z][A-Za-z0-9\s\-]{1,40}):')
        matches = list(marker_pattern.finditer(text))

        for idx, match in enumerate(matches):
            label = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            phrase = text[start:end].strip()
            if phrase:
                fallback_sections.append((label, phrase))

        return fallback_sections

    def _looks_like_label(self, label: str) -> bool:
        normalized = re.sub(r'\s+', ' ', label).strip().lower()
        if not normalized:
            return False
        return len(normalized) <= 40 and any(ch.isalpha() for ch in normalized)

    def _canonicalize_body_part(self, raw_label: str) -> Optional[str]:
        """Map raw section label to explicit morphology keys required by the pipeline."""
        label = re.sub(r'[_\-]+', ' ', raw_label.lower())
        label = re.sub(r'\s+', ' ', label).strip()

        if label in self.body_part_aliases:
            return self.body_part_aliases[label]

        maxilliped_match = re.match(r'^maxilliped\s+([0-9ivx]+)$', label)
        if maxilliped_match:
            idx = self._to_int_index(maxilliped_match.group(1))
            if idx in {2, 3}:
                return f'maxilliped_{idx}'

        pereopod_match = re.match(r'^pereopod\s+([0-9ivx]+)$', label)
        if pereopod_match:
            idx = self._to_int_index(pereopod_match.group(1))
            if idx in {1, 2, 3, 4, 5}:
                return f'pereopod_{idx}'

        antenna_match = re.match(r'^antenna\s+([0-9ivx]+)$', label)
        if antenna_match:
            idx = self._to_int_index(antenna_match.group(1))
            if idx == 1:
                return 'antenna_1'

        return None

    def _to_int_index(self, value: str) -> Optional[int]:
        roman_to_int = {
            'i': 1,
            'ii': 2,
            'iii': 3,
            'iv': 4,
            'v': 5,
        }
        value = value.strip().lower()
        if value.isdigit():
            return int(value)
        return roman_to_int.get(value)

    def _clean_phrase(self, phrase: str) -> str:
        cleaned = re.sub(r'\s+', ' ', phrase).strip()
        return cleaned

    def _infer_view(self, illustration_path: Optional[str]) -> str:
        if not illustration_path:
            return 'lateral'

        filename = illustration_path.lower()
        if 'lateral' in filename:
            return 'lateral'
        if 'dorsal' in filename:
            return 'dorsal'
        if 'ventral' in filename:
            return 'ventral'
        if 'anterior' in filename:
            return 'anterior'
        if 'posterior' in filename:
            return 'posterior'
        return 'lateral'

    def _count_clip_tokens(self, text: str) -> int:
        try:
            from transformers import CLIPTokenizer
            if self._clip_tokenizer is None:
                self._clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
                self._clip_tokenizer.model_max_length = 1000000
            token_ids = self._clip_tokenizer.encode(text)
            return len(token_ids)
        except Exception:
            return len(text.split())

    def _constraint_line(self, key: str, value: str) -> str:
        label = self.constraint_display_labels.get(key, key.replace('_', ' ').title())
        return f"{label}: {value}"

    def _compress_scientific_phrase(self, phrase: str) -> str:
        """
        Compress scientific phrase while preserving taxonomic meaning.
        
        Removes filler words but preserves complete short sentences.
        Never creates sentence fragments.
        """
        compressed = phrase
        
        # Remove redundant articles (but preserve sentence structure)
        compressed = re.sub(r'\bthe\s+', '', compressed, flags=re.IGNORECASE)
        compressed = re.sub(r'\b(a|an)\s+(?!little|few)', '', compressed, flags=re.IGNORECASE)
        
        # Simplify "is/are X" to just "X"
        compressed = re.sub(r'\b(is|are)\s+', '', compressed, flags=re.IGNORECASE)
        
        # Remove "which is/are/has"
        compressed = re.sub(r'\bwhich\s+(is|are|has)\s+', '', compressed, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        # Remove trailing punctuation
        compressed = compressed.rstrip('.,;')
        
        return compressed

    def _build_prompt_lines(
        self,
        species_name: str,
        view: str,
        constraints: Dict[str, str],
        max_clip_tokens: int
    ) -> Dict[str, Any]:
        # Base emphasizing photorealistic rendering while maintaining anatomical fidelity
        base_lines = [
            f"{species_name} {view} view.",
            "High-quality professional photograph of specimen, hyper-realistic.",
            "Preserve exact body proportions and segment ratios from the reference illustration.",
        ]
        closing_line = 'Detailed exoskeleton texture, natural museum lighting, photographic realism, biological specimen photography.'

        ordered_keys = [key for key in self.constraint_order if key in constraints and constraints[key]]
        if not ordered_keys:
            prompt_lines = list(base_lines)
            with_closing = '\n'.join(prompt_lines + [closing_line])
            if self._count_clip_tokens(with_closing) <= max_clip_tokens:
                prompt_lines.append(closing_line)

            prompt = '\n'.join(prompt_lines)
            token_count = self._count_clip_tokens(prompt)
            return {
                'prompt': prompt,
                'full_prompt_before_compression': prompt,
                'token_count': token_count,
                'truncated': token_count > max_clip_tokens,
                'view': view,
                'included_body_parts': [],
                'dropped_body_parts': [],
                'full_scientific_phrases_used': True,
                'summarization_applied': False,
                'truncation_strategy': 'none',
                'compression_strategy': 'none',
            }

        # Try to include ALL morphological details with full phrases
        all_detail_lines = [self._constraint_line(key, constraints[key]) for key in ordered_keys]
        full_lines = base_lines + ['Morphology:'] + all_detail_lines + [closing_line]
        full_prompt = '\n'.join(full_lines)
        full_token_count = self._count_clip_tokens(full_prompt)

        if full_token_count <= max_clip_tokens:
            return {
                'prompt': full_prompt,
                'full_prompt_before_compression': full_prompt,
                'token_count': full_token_count,
                'truncated': False,
                'view': view,
                'included_body_parts': ordered_keys,
                'dropped_body_parts': [],
                'full_scientific_phrases_used': True,
                'summarization_applied': False,
                'truncation_strategy': 'none',
                'compression_strategy': 'none',
            }

        # Over budget - try compressing morphological details while preserving scientific meaning
        # Store full prompt before compression for debugging
        full_prompt_before_compression = full_prompt
        
        compressed_lines = []
        for key in ordered_keys:
            compressed = self._compress_scientific_phrase(constraints[key])
            compressed_lines.append(f"{self.constraint_display_labels[key]}: {compressed}.")
        
        compressed_full_lines = base_lines + ['Morphology:'] + compressed_lines + [closing_line]
        compressed_prompt = '\n'.join(compressed_full_lines)
        compressed_token_count = self._count_clip_tokens(compressed_prompt)

        if compressed_token_count <= max_clip_tokens:
            return {
                'prompt': compressed_prompt,
                'token_count': compressed_token_count,
                'truncated': False,
                'view': view,
                'included_body_parts': ordered_keys,
                'dropped_body_parts': [],
                'full_scientific_phrases_used': True,
                'summarization_applied': True,  # Compression applied but meaning preserved
                'truncation_strategy': 'compress_scientific_phrases',
            }

        # Still over budget after compression - select complete clauses for priority parts
        # Strategy: Ensure essential priority parts (carapace, pseudorostrum, pereopod_2, uropod)
        # get at least one clause each before adding more detail
        
        essential_priority_order = [key for key in self.essential_priority_keys if key in ordered_keys]
        other_priority_order = [key for key in self.priority_keys if key in ordered_keys and key not in essential_priority_order]
        non_priority_order = [key for key in ordered_keys if key not in self.priority_keys]

        selected_lines: List[str] = []
        included_keys: List[str] = []

        # Phase 1a: Allocate FIRST clause from each essential priority part
        # This ensures carapace, pseudorostrum, pereopod_2, uropod all get representation
        for key in essential_priority_order:
            compressed = self._compress_scientific_phrase(constraints[key])
            
            # Split into clauses (by semicolon or comma)
            if ';' in compressed:
                clauses = [c.strip() for c in compressed.split(';') if c.strip()]
            else:
                clauses = [c.strip() for c in compressed.split(',') if c.strip()]
            
            # Take ONLY the first clause to ensure all essential parts fit
            if clauses:
                first_clause = clauses[0]
                test_line = f"{self.constraint_display_labels[key]}: {first_clause}."
                trial_lines = base_lines + ['Morphology:'] + selected_lines + [test_line] + [closing_line]
                trial_prompt = '\n'.join(trial_lines)
                
                if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                    selected_lines.append(test_line)
                    included_keys.append(key)
                else:
                    # If even first clause doesn't fit, we're truly out of budget
                    # But try to continue with remaining parts
                    pass
        
        # Phase 1b: Try to add MORE clauses to essential priority parts if budget allows
        for key in essential_priority_order:
            if key not in included_keys:
                continue  # Skip if not already included
            
            compressed = self._compress_scientific_phrase(constraints[key])
            if ';' in compressed:
                clauses = [c.strip() for c in compressed.split(';') if c.strip()]
            else:
                clauses = [c.strip() for c in compressed.split(',') if c.strip()]
            
            # We already have the first clause, try adding more
            if len(clauses) > 1:
                # Find the existing line for this key
                existing_line_idx = None
                for idx, line in enumerate(selected_lines):
                    if line.startswith(f"{self.constraint_display_labels[key]}:"):
                        existing_line_idx = idx
                        break
                
                if existing_line_idx is not None:
                    # Try to expand with more clauses
                    for num_clauses in range(2, len(clauses) + 1):
                        selected_clauses = clauses[:num_clauses]
                        clause_text = '; '.join(selected_clauses)
                        test_line = f"{self.constraint_display_labels[key]}: {clause_text}."
                        
                        # Replace the existing line temporarily
                        test_lines = selected_lines[:existing_line_idx] + [test_line] + selected_lines[existing_line_idx+1:]
                        trial_lines = base_lines + ['Morphology:'] + test_lines + [closing_line]
                        trial_prompt = '\n'.join(trial_lines)
                        
                        if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                            selected_lines[existing_line_idx] = test_line
                        else:
                            break  # Can't add more clauses, keep what we have
        
        # Phase 2: Add other priority parts with clause-based selection
        for key in other_priority_order:
            compressed = self._compress_scientific_phrase(constraints[key])
            
            if ';' in compressed:
                clauses = [c.strip() for c in compressed.split(';') if c.strip()]
            else:
                clauses = [c.strip() for c in compressed.split(',') if c.strip()]
            
            # Try to fit complete clauses starting from the most important
            added = False
            for num_clauses in range(len(clauses), 0, -1):
                selected_clauses = clauses[:num_clauses]
                clause_text = '; '.join(selected_clauses) if ';' in compressed else ', '.join(selected_clauses)
                test_line = f"{self.constraint_display_labels[key]}: {clause_text}."
                trial_lines = base_lines + ['Morphology:'] + selected_lines + [test_line] + [closing_line]
                trial_prompt = '\n'.join(trial_lines)
                
                if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                    selected_lines.append(test_line)
                    included_keys.append(key)
                    added = True
                    break
            
            if not added:
                continue
        
        # Phase 3: Add non-priority parts with complete compressed phrases
        for key in non_priority_order:
            compressed = self._compress_scientific_phrase(constraints[key])
            test_line = f"{self.constraint_display_labels[key]}: {compressed}."
            trial_lines = base_lines + ['Morphology:'] + selected_lines + [test_line] + [closing_line]
            trial_prompt = '\n'.join(trial_lines)

            if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                selected_lines.append(test_line)
                included_keys.append(key)

        final_lines = list(base_lines)
        if selected_lines:
            final_lines += ['Morphology:'] + selected_lines

        # Always include closing line (we reserved budget for it)
        final_lines.append(closing_line)

        final_prompt = '\n'.join(final_lines)
        final_token_count = self._count_clip_tokens(final_prompt)
        dropped_keys = [key for key in ordered_keys if key not in included_keys]

        return {
            'prompt': final_prompt,
            'full_prompt_before_compression': full_prompt_before_compression,
            'token_count': final_token_count,
            'truncated': len(dropped_keys) > 0,
            'view': view,
            'included_body_parts': included_keys,
            'dropped_body_parts': dropped_keys,
            'full_scientific_phrases_used': True,
            'summarization_applied': True,
            'truncation_strategy': 'select_complete_clauses_and_drop_low_priority',
            'compression_strategy': 'preserve_complete_sentences',
        }

    def build_morphology_prompt(
        self,
        species_name: str,
        constraints: Dict[str, str],
        illustration_path: Optional[str] = None,
        max_clip_tokens: int = 77,
        return_metadata: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Build a structured morphological prompt from parsed constraints.

        Args:
            species_name: Name of the species (e.g., "Campylaspis aculeata")
            constraints: Dictionary of morphological constraints (body part -> description)
            illustration_path: Path to illustration for view extraction
            max_clip_tokens: Maximum CLIP token budget (default 77)
            return_metadata: If True, return prompt metadata including token count

        Returns:
            Prompt string, or metadata dictionary if return_metadata=True
        """
        view = self._infer_view(illustration_path)
        prompt_result = self._build_prompt_lines(
            species_name=species_name,
            view=view,
            constraints=constraints,
            max_clip_tokens=max_clip_tokens,
        )

        if return_metadata:
            return prompt_result
        return prompt_result['prompt']
    
    def validate_constraints(self, constraints: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate morphological constraints for completeness and consistency.

        Args:
            constraints: Dictionary of morphological constraints

        Returns:
            Validation results with warnings and suggestions
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'suggestions': []
        }
        
        essential_parts = ['carapace', 'pseudorostrum', 'pereopod_2', 'uropod']
        missing_parts = []
        
        for part in essential_parts:
            if part not in constraints or not constraints[part]:
                missing_parts.append(part)
        
        if missing_parts:
            validation_results['warnings'].append(
                f"Missing descriptions for essential body parts: {', '.join(missing_parts)}"
            )
            validation_results['suggestions'].append(
                "Consider adding descriptions for all essential morphological features"
            )
        
        empty_parts = [part for part, description in constraints.items() if not description]
        if empty_parts:
            validation_results['warnings'].append(
                f"Empty constraints for body parts: {', '.join(empty_parts)}"
            )
        
        if not constraints:
            validation_results['valid'] = False
            validation_results['warnings'].append("No morphological constraints found")
        
        return validation_results


def parse_morphology_from_text(description_text: str) -> Dict[str, str]:
    """
    Convenience function to parse morphological constraints from text.
    
    Args:
        description_text: Raw taxonomic description text
        
    Returns:
        Dictionary of morphological constraints
    """
    parser = MorphologyParser()
    return parser.parse_taxonomic_description(description_text)


def build_morphology_prompt(
    species_name: str,
    constraints: Dict[str, str],
    illustration_path: Optional[str] = None,
    max_clip_tokens: int = 77,
    return_metadata: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Convenience function to build morphology prompt from constraints.
    
    Args:
        species_name: Name of the species
        constraints: Dictionary of morphological constraints
        illustration_path: Path to illustration for view extraction
        max_clip_tokens: Maximum CLIP token budget (default 77)
        return_metadata: If True, return prompt metadata including token count

    Returns:
        Prompt string, or metadata dictionary if return_metadata=True
    """
    parser = MorphologyParser()
    return parser.build_morphology_prompt(
        species_name=species_name,
        constraints=constraints,
        illustration_path=illustration_path,
        max_clip_tokens=max_clip_tokens,
        return_metadata=return_metadata,
    )