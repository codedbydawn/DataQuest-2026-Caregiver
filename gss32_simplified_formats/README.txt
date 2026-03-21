GSS 2018 Caregiving/Care Receiving - simplified machine-friendly files

Files included
1. codebook_variables.jsonl
   - One JSON object per variable (572 total)
   - Fields: variable_name, length, position, question_name, concept, question_text,
     universe, note, source, answer_categories
   - answer_categories is an array of objects with: label, code, frequency,
     weighted_frequency, percent

2. codebook_variables.csv
   - Flat CSV with one row per variable
   - answer_categories_json contains the category array as JSON text

3. codebook_answer_categories.csv
   - One row per answer category (4139 total)
   - Good for quick filtering/searching

4. user_guide_clean.txt
   - Plain text extraction of the user guide, organized by page

5. user_guide_pages.json
   - JSON array with page number + cleaned text

6. user_guide_sections.json
   - Basic section index from the guide TOC

Notes
- These files were extracted from the uploaded PDFs:
  * c32pumf_cgcr_codebook_NEW_E.pdf
  * C32PUMF_Guide_E.pdf
- The codebook PDF is a data dictionary / documentation, not the raw survey microdata.
- In the user guide, Statistics Canada notes that reserve codes include:
  * 6 = Valid skip
  * 9 = Not stated
  and that social survey variable names are 8 characters or less.
- The guide also notes that person-level estimates should generally use WGHT_PER.

Potential caveats
- PDF extraction can introduce minor line-break artifacts.
- Long answer labels that wrapped in the PDF were flattened into single lines.
- Numeric fields are kept as text in CSV/JSON where preserving the original code format matters.
