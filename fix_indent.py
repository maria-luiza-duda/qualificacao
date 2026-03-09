#!/usr/bin/env python3
"""Fix indentation in morphology_parser.py"""

file_path = "src/campylaspis/generation/morphology_parser.py"

# Read the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix specific lines with indentation issues
# Line 382-383 (0-indexed would be 381-382)
# Line 387 (0-indexed would be 386)
# Line 398 (0-indexed would be 397)

fixed_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    
    # Fix line 382-383
    if line_num in [382, 383]:
        # Remove 4 extra spaces of indentation
        if line.startswith('            #'):
            fixed_lines.append(line[4:])
        else:
            fixed_lines.append(line)
    # Fix line 387
    elif line_num == 387:
        # Remove 8 extra spaces of indentation
        if line.startswith('                trial_lines'):
            fixed_lines.append(line[8:])
        else:
            fixed_lines.append(line)
    # Fix line 398
    elif line_num == 398:
        # Remove 4 extra spaces of indentation
        if line.startswith('            #'):
            fixed_lines.append(line[4:])
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open(file_path, 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation errors")
