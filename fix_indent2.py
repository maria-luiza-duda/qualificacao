#!/usr/bin/env python3
"""Fix all indentation issues in morphology_parser.py section selection"""

file_path = "src/campylaspis/generation/morphology_parser.py"

with open(file_path, 'r') as f:
    content = f.read()

# The problematic section starts around line 373
# Replace the entire problematic section with correct indent

old_section = '''        selected_lines: List[str] = []
        included_keys: List[str] = []

            # Reserve tokens for closing line (critical for realistic rendering)
        # by including it in trial during selection

        for key in candidate_order:
            compressed = self._compress_scientific_phrase(constraints[key])
            test_line = f"{self.constraint_display_labels[key]}: {compressed}"
        trial_lines = base_lines + ['Morphology:'] + selected_lines + [test_line] + [closing_line]
            trial_prompt = '\\n'.join(trial_lines)

            if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                selected_lines.append(test_line)
                included_keys.append(key)

        final_lines = list(base_lines)
        if selected_lines:
            final_lines += ['Morphology:'] + selected_lines

        # Always include closing line (we reserved budget for it)
            final_lines.append(closing_line)'''

new_section = '''        selected_lines: List[str] = []
        included_keys: List[str] = []

        # Reserve tokens for closing line (critical for realistic rendering)
        # by including it in trial during selection

        for key in candidate_order:
            compressed = self._compress_scientific_phrase(constraints[key])
            test_line = f"{self.constraint_display_labels[key]}: {compressed}"
            trial_lines = base_lines + ['Morphology:'] + selected_lines + [test_line] + [closing_line]
            trial_prompt = '\\n'.join(trial_lines)

            if self._count_clip_tokens(trial_prompt) <= max_clip_tokens:
                selected_lines.append(test_line)
                included_keys.append(key)

        final_lines = list(base_lines)
        if selected_lines:
            final_lines += ['Morphology:'] + selected_lines

        # Always include closing line (we reserved budget for it)
        final_lines.append(closing_line)'''

content = content.replace(old_section, new_section)

with open(file_path, 'w') as f:
    f.write(content)

print("Fixed all indentation issues")
