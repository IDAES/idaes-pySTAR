import csv
from expression_to_tree import parse_math, tree_depth

# Read the original CSV file
input_file = "FeynmanEquations.csv"
output_file = "Feynman_depths.csv"

# Open input and output files with explicit encoding
with open(input_file, "r", encoding="utf-8-sig") as infile:
    # Read all lines and strip BOM/whitespace from header
    lines = infile.readlines()

# Process the CSV with cleaned header
reader = csv.DictReader(lines)

# Clean column names to remove any hidden characters
cleaned_fieldnames = [field.strip() for field in reader.fieldnames]

# Define the output columns
output_fieldnames = ["Filename", "Formula", "# variables", "depth"]

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
    writer.writeheader()

    # Process each row
    for row_num, row in enumerate(reader, start=2):
        # Create cleaned row dict
        cleaned_row = {
            cleaned_fieldnames[i]: value for i, value in enumerate(row.values())
        }

        filename = cleaned_row.get("Filename", "").strip()
        formula = cleaned_row.get("Formula", "").strip()
        num_variables = cleaned_row.get("# variables", "").strip()

        # Skip empty rows
        if not filename or not formula:
            continue

        try:
            # Parse the expression and calculate depth
            expr = parse_math(formula)
            depth = tree_depth(expr)

            # Write to output file
            writer.writerow(
                {
                    "Filename": filename,
                    "Formula": formula,
                    "# variables": num_variables,
                    "depth": depth,
                }
            )

            print(f"Processed {filename}: depth = {depth}")

        except Exception as e:
            print(f"Error processing row {row_num} ({filename}): {e}")
            continue

print(f"\nResults saved to {output_file}")
